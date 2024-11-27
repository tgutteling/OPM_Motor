# Here we import the raw OPM-MEG data and
# > remove break periods
# > remove spikes
# > Remove amplitude violations
# > run ICA
# > Check and reject components based on visual inspection (continuous and epoched)
# > Remove artifact components
# > filter cleaned data
# > epoch cleaned data
# > downsample to 1Khz
# > Clean epoched data using regression of the Reference sensors
# > Calculate evoked field

#Preamble
#from logging import warning
import os.path as op
from os import listdir
import sys
## MNE toolbox
import mne
import numpy as np
from itertools import groupby
import autoreject as ar
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from dtw import *

#Script folder
sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
import mag4health_5sens as m4he

import matplotlib.pyplot as plt 
plt.ion()

#Set up data for import
data_path = '/sps/cermep/opm/NEW_MEG/HV'

#Get subject directories
dirs = [int(d) for d in listdir(data_path) if op.isdir(op.join(data_path, d)) and d.isnumeric()]
dirs.sort()

#Let user select subject
print('Available Subjects: ')
for d in range(len(dirs)):          
        print(str(dirs[d]).zfill(2))
    
input_given=False
while not input_given:
    inp = input('Select Subject: ') #Ask for user input    
    if inp.isdecimal() and np.any(np.isin(dirs,int(inp))):
        print('Subject '+inp+' selected')
        subject=inp.zfill(2)        
        input_given=True
    else:
        print('Incorrect input, please try again')

print('Retreiving OPM sessions..')    
OPM_files=[d for d in listdir(op.join(data_path, subject)) if 'OPM' in d and 'joyst' in d and not 'noTrig'  in d and op.isdir(op.join(data_path, subject,d))]
print('Found ' +str(len(OPM_files))+ ' OPM data directories')
OPM_files.sort()


#Task parameters 
#Triggers (names):
    #'13'     1 - start/end block
    #'end'    8 - end start block
    #'2'      2 - fixation [1s - 2s]
    #'3'      3 - dot display <-- visual stim [2s]
    #'4'      4 - delay [0.45s - 1s]
    #'5'      5 - target [0-1s]
    #'6'      6 - feedback [1.5-4s]
    #'7'      7 - feedback <--motor [2-2.9s]

#Epoching:
    #Visual:  [-1 2]
    #Motor:   [-1 1.5]

tminVIS,tmaxVIS = -1, 2 #time before and after trigger
tminMOT,tmaxMOT = -1, 1.5 #time before and after trigger
bl_min,bl_max=-.8,-.3 #baseline start and end (None for either first or last sample)
highpass,lowpass = 1,100 #bandpass frequencies
line_freq=[50] #notch filter frequency(/ies)
final_samp_rate=1000 #Output sampling rate
use_previous_rejection=0 #load rejection parameters from previous dataset, for consistency when reanalysing with different parameters
use_autoreject=1

all_raw=[]
for i_run in OPM_files:
    print('Loading ' +i_run)
    file_name=op.join(data_path, subject, i_run, 'data', 'single_matrix.mat') 
    
    if op.exists(file_name):
        raw=m4he.read_mag4health_raw(file_name)
    
        events, event_dict=mne.events_from_annotations(raw)      
        
        #Let's remove periods without stimulation events (startup, end, breaks)
        break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        events=events,
        min_break_duration=9,  # consider segments of at least 5 s duration
        t_start_after_previous=4,  # buffer time after last event, carefull of edge effects
        t_stop_before_next=4  # stop annotation 4 s before beginning of next one
        )
        raw.set_annotations(raw.annotations + break_annots)  #Mark breaks in raw data
        
        #Mark _z direction channels as bad
        bads=mne.pick_channels_regexp(raw.info['ch_names'],'^.*z')
        raw.info['bads'].extend([raw.info['ch_names'][x] for x in bads])
        
        #Remove spikes from data
        #Reject by amplitude, use standard deviation of the raw data
        raw_tmp=raw.copy().resample(500).filter(l_freq=1, h_freq=100)    
        ref_chans=mne.pick_channels_regexp(raw.info['ch_names'],'^ZC*')
        exclude=bads+ref_chans
        goods=list(set(range(0,len(raw.ch_names)))-set(exclude))        
        dat,times=raw_tmp.copy().crop(tmin=15,tmax=raw_tmp.times[-1]-15).get_data(picks=goods,reject_by_annotation='omit',return_times=True) #raw data, just for setting threshold                        
        n_sds=5 #how many SD to set as threshold
        spike_annots, spike_bads = mne.preprocessing.annotate_amplitude(
            raw_tmp,
            peak=np.median(dat)+np.std(dat)*n_sds
            )        
        raw.set_annotations(raw.annotations + spike_annots) #add these annotations to the raw data
        
        
        #spikes get removed fine, but slower high amplitude artifacts do not get picked up
        #This is a first pass rejection to remove gross artifacts. This step will be repeated for the whole timeseries as well        
        raw_tmp.annotations.onset=raw_tmp.annotations.onset-raw_tmp.first_time #see note                
        raw_tmp.set_annotations(raw_tmp.annotations + spike_annots) #add these annotations to the raw data
        dat,times=raw_tmp.get_data(picks=goods,reject_by_annotation='NaN',return_times=True) #get raw data and times
        dat_crop=raw_tmp.copy().crop(tmin=15,tmax=raw_tmp.times[-1]-15).get_data(picks=goods,reject_by_annotation='omit')
        n_sds=5 #how many SD to set as threshold
        thres=np.median(dat_crop)+np.std(dat_crop)*n_sds #set the rejection threshold using SD
        bad=np.abs(dat)>thres #create bool array of threshold violations
        bad=[bad[:,i].any() for i in range(bad.shape[1])] #collapse across channels
        #Now place a 20ms sliding window in the true ranges 
        n_samp = round(raw_tmp.info['sfreq']*.01)    
        i=0
        while i<len(bad):
            if bad[i]:
                if i < n_samp:
                    bad[:i+n_samp] = np.full(len(bad[:i+n_samp]),True)
                    i+=n_samp+1                
                elif i+n_samp > len(bad):
                    bad[i-n_samp:] = np.full(len(bad[i-n_samp:]),True)                
                    break
                else:
                    bad[i-n_samp:i+n_samp] = np.full(len(bad[i-n_samp:i+n_samp]),True)                
                    i+=n_samp+1
            else:
                i+=1                        
    
        #To create annotations from this, we need to know onset times and durations
        #bad_times = times[bad]    
        onset=[]
        duration=[]
        description=[]
        i = 0
        for val, g in groupby(bad):
            l=len(list(g))
            if val:       
                onset.append(times[i]) 
                if i+l>=len(times):
                    duration.append(times[-1]-times[i])
                else:
                    duration.append(times[i+l]-times[i])
                description.append('BAD_amplitude')
            i += l
        
        #Create annotations and add to data
        amplitude_annots=mne.Annotations(onset=onset, duration=duration,description=description)
        raw.annotations.onset=raw.annotations.onset-raw.first_time #see note        
        raw.set_annotations(raw.annotations + amplitude_annots)
    
    all_raw.append(raw) #add annotated raw session to the whole
    
raw=mne.concatenate_raws(all_raw, on_mismatch='warn')    

#get events from concatenated RAW
events, event_dict=mne.events_from_annotations(raw)
       
#Subject 8 has a timing synchronization error between channesl in the first 560s. We need to remove this.
if np.any(np.isin(int(subject), [8])):
    raw.crop(tmin=560,tmax=None)
    raw_tmp=mne.io.RawArray(raw.get_data(),raw.info)
    raw_tmp._annotations=raw.annotations
    raw_tmp._annotations.onset=raw_tmp._annotations.onset-raw.first_time
    raw=raw_tmp    
    events, event_dict=mne.events_from_annotations(raw)


#Now that we have collected all session and removed all gross artefacts from the data, we can use the data for ICA
print('Creating downsampled, filtered RAW data for ICA')
raw_ica=raw.copy().resample(500).filter(l_freq=1, h_freq=100)    
raw_ica.notch_filter(freqs=[50,60,100],n_jobs=-1)

#Compute ICA
print('Computing ICA')
ica = mne.preprocessing.ICA(method='fastica',random_state=42)
ica.fit(raw_ica)
        
if not use_previous_rejection:
    #Visually inspect components
    ica.plot_sources(raw_ica)
    
    #for comparison, use epoched data    
    raw_ep=raw.copy().filter(l_freq=1, h_freq=100)
    event_id_VIS={'CODE3\n' : event_dict['CODE3\n']}                  
    event_id_MOT={'CODE7\n' : event_dict['CODE7\n']}  
    tmp_epochs=mne.Epochs(raw_ep,events,event_id=event_id_MOT,tmin=tminMOT,tmax=tmaxMOT,preload=True)
    tmp_epochs.resample(500)
    ica.plot_sources(tmp_epochs)
    
#BAD components should now be marked   
ICA_reject = {
    '01' : [2,3],
    '02' : [0],
    '04' : [9],
    '06' : [0,1],
    '07' : [],
    '08' : [],
    '09' : [0],
    '10' : [0,1],
    '12' : [0],
    '14' : [3],
    '15' : [2],
    '16' : [1],
    '17' : [0],
    '18' : [0],
    '19' : [0],
    '21' : [0],
    '23' : [0],
    '24' : [2]
    }

#Bandpass filter
print('Wide Bandpass filter')
raw_filt=raw.copy().filter(l_freq=highpass, h_freq=lowpass,n_jobs=-1) # bandpass
raw_filt.notch_filter(freqs=[50,60,100])

#Apply ICA to raw, unfiltered data
if use_previous_rejection:
    raw_filt=ica.apply(raw_filt,exclude=ICA_reject[subject])    
else:
    raw_filt=ica.apply(raw_filt)       

#Define VISual and MOTor event triggers
event_id_VIS={'CODE3\n' : event_dict['CODE3\n']}                  
event_id_MOT={'CODE7\n' : event_dict['CODE7\n']}                  
events_VIS, event_dict_VIS=mne.events_from_annotations(raw, event_id=event_id_VIS)
events_MOT, event_dict_MOT=mne.events_from_annotations(raw, event_id=event_id_MOT)

#Create epochs; We'll use the visual epochs as the reference for autoregression
epochsVIS=mne.Epochs(raw_filt,events_VIS,event_id=event_id_VIS,tmin=tminVIS,tmax=tmaxVIS,preload=True)
raw_filt._annotations=mne.Annotations(onset=[], duration=[],description='') #the annotation may be too stringent for uncorrected motor data, we'll remove artifacts later
epochsMOT=mne.Epochs(raw_filt,events_MOT,event_id=event_id_MOT,tmin=tminMOT,tmax=tmaxMOT,preload=True)

#now resample to output sampling rate
epochsVIS.resample(final_samp_rate)
epochsMOT.resample(final_samp_rate)
    
#add bad channels back in for diagnostics
epochsVIS.info['bads']=[]
epochsMOT.info['bads']=[]

#fancy pantsy Reference regression
#Visual
picks_x=mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^L.*x')+mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^R.*x')
picks_y=mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^L.*y')+mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^R.*y')
picks_z=mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^L.*z')+mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^R.*z')
scalp_chans=picks_x+picks_y+picks_z
scalp_chans.sort()
ref_chans=mne.pick_channels_regexp(epochsVIS.info['ch_names'],'^Z.*')

ref_filt=epochsVIS.copy().pick(ref_chans).filter(l_freq=0,h_freq=10,verbose='error').get_data() #only low frequency fluctuation
ref_unfilt=epochsVIS.copy().pick(ref_chans).get_data()
ep_vis_dat=epochsVIS.get_data()
ep_vis_dat_regr=np.zeros(np.shape(ep_vis_dat))
print('Doing fancy DTW reference regression (visual data)')
ch_cnt=1
for ch in scalp_chans: #per channel
    print('(VIS) Denoising channel '+epochsVIS.ch_names[ch]+  ' ' +str(ch_cnt)+ '/' +str(len(scalp_chans)))
    ch_cnt=ch_cnt+1
    cur_ch=ep_vis_dat[:,ch,:]    
                
    for tr in range(np.shape(cur_ch)[0]):
        
        #select data
        trial=cur_ch[tr,]
        cur_ref=ref_filt[tr,:,:]
                                              
        #fit reference to trial data
        reg = LinearRegression().fit(cur_ref.T, trial)
        tmp=np.sum(reg.coef_*cur_ref.T,axis=1)
        nowarp=np.sum(reg.coef_*cur_ref.T,axis=1)
        
        # #optimize alignment with dynamic time warp
        alignment_z = dtw(tmp,trial,step_pattern=symmetricP2,window_type=sakoeChibaWindow,window_args={'window_size': 50} )
        tmp[0:len((warp(alignment_z,index_reference=False)))]=tmp[(warp(alignment_z,index_reference=False))]
                
        # #remove high frequencies introduced by the warping
        tmp_filt=mne.filter.filter_data(tmp,1000,l_freq=0,h_freq=10,verbose='error')
        
        # #after warp, fit may be improved
        regr=np.ones([len(trial),2]) #create regressors, intercept and evoked
        regr[:,1]=tmp_filt #add evoked
        reg = LinearRegression().fit(regr, trial)
        ref_warp=(regr[:,1]*reg.coef_[1])
                
        #only apply warping if it leads to lower RMS
        if np.sum(np.sqrt((trial-nowarp)**2))<=np.sum(np.sqrt((trial-ref_warp)**2)):
            ep_vis_dat_regr[tr,ch,] = trial-nowarp
        else:
            ep_vis_dat_regr[tr,ch,] = trial-ref_warp
                   
#Put cleaned data in new data structure
epochsVIS_regr=deepcopy(epochsVIS)
epochsVIS_regr._data=ep_vis_dat_regr
epochsVIS_regr.pick(scalp_chans)
epochsVIS_regr.apply_baseline(baseline=(bl_min,bl_max))  

#Motor
picks_x=mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^L.*x')+mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^R.*x')
picks_y=mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^L.*y')+mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^R.*y')
picks_z=mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^L.*z')+mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^R.*z')
scalp_chans=picks_x+picks_y+picks_z
scalp_chans.sort()
ref_chans=mne.pick_channels_regexp(epochsMOT.info['ch_names'],'^Z.*')

ref_filt=epochsMOT.copy().pick(ref_chans).filter(l_freq=0,h_freq=10,verbose='error').get_data() #only low frequency fluctuation
ep_mot_dat=epochsMOT.get_data()
ep_mot_dat_regr=np.zeros(np.shape(ep_mot_dat))
print('Doing fancy DTW reference regression (motor data).')
ch_cnt=1
for ch in scalp_chans: #per channel
    print('(MOT) Denoising channel '+epochsMOT.ch_names[ch]+  ' ' +str(ch_cnt)+ '/' +str(len(scalp_chans)))
    ch_cnt=ch_cnt+1
    cur_ch=ep_mot_dat[:,ch,:]
        
    for tr in range(np.shape(cur_ch)[0]):
        
        #select data
        trial=cur_ch[tr,]
        cur_ref=ref_filt[tr,:,:]            
        
        #fit reference to trial data
        reg = LinearRegression().fit(cur_ref.T, trial)
        tmp=np.sum(reg.coef_*cur_ref.T,axis=1)
        nowarp=np.sum(reg.coef_*cur_ref.T,axis=1)
        
        #optimize alignment with dynamic time warp
        alignment_z = dtw(tmp,trial,step_pattern=symmetricP2,window_type=sakoeChibaWindow,window_args={'window_size': 50} ) #close, seems fastest        
        tmp[0:len((warp(alignment_z,index_reference=False)))]=tmp[(warp(alignment_z,index_reference=False))]
                
        #remove high frequencies introduced by the warping
        tmp_filt=mne.filter.filter_data(tmp,1000,l_freq=0,h_freq=10,verbose='error')
        
        #after warp, fit may be improved
        regr=np.ones([len(trial),2]) #create regressors, intercept and evoked
        regr[:,1]=tmp_filt #add evoked
        reg = LinearRegression().fit(regr, trial)
        ref_warp=(regr[:,1]*reg.coef_[1])                                             
        
        #only apply warping if it leads to lower RMS
        if np.sum(np.sqrt((trial-nowarp)**2))<=np.sum(np.sqrt((trial-ref_warp)**2)):
            ep_mot_dat_regr[tr,ch,] = trial-nowarp
        else:
            ep_mot_dat_regr[tr,ch,] = trial-ref_warp

#Put cleaned data in new data structure
epochsMOT_regr=deepcopy(epochsMOT)
epochsMOT_regr._data=ep_mot_dat_regr
epochsMOT_regr.pick(scalp_chans)
epochsMOT_regr.apply_baseline(baseline=(bl_min,bl_max))     
    

#Do another round of spike detection after preprocessing, seems some like to hang around
#remove them spikeys
picks_x=mne.pick_channels_regexp(epochsVIS_regr.info['ch_names'],'^L.*x')+mne.pick_channels_regexp(epochsVIS_regr.info['ch_names'],'^R.*x')
picks_y=mne.pick_channels_regexp(epochsVIS_regr.info['ch_names'],'^L.*y')+mne.pick_channels_regexp(epochsVIS_regr.info['ch_names'],'^R.*y')
tmp_datVIS_regr=epochsVIS_regr.get_data(picks=picks_x+picks_y)
tmp_datMOT_regr=epochsMOT_regr.get_data(picks=picks_x+picks_y)
VISregrp2p=np.max(np.abs((np.diff(tmp_datVIS_regr,axis=2))),axis=2) #get the absolute sample-to-sample difference per sensor and trial
MOTregrp2p=np.max(np.abs((np.diff(tmp_datMOT_regr,axis=2))),axis=2)

#define threshold and drop them epochs
thresh=np.median(VISregrp2p)+5*np.std(VISregrp2p)
drop_mask=np.any(VISregrp2p>thresh,axis=1)
print('Visual data (regressed) spike detection: Found ' +str(len(np.where(drop_mask)[0]))+ ' peak(s), removing..')
epochsVIS_regr.drop(drop_mask)

thresh=np.median(MOTregrp2p)+5*np.std(MOTregrp2p)
drop_mask=np.any(MOTregrp2p>thresh,axis=1)
print('Motor data (regressed) spike detection: Found ' +str(len(np.where(drop_mask)[0]))+ ' peak(s), removing..')
epochsMOT_regr.drop(drop_mask)

#fancy DTW evoked removal

#VISUAL
ev_filt=epochsVIS_regr.copy().filter(l_freq=0,h_freq=10,verbose='error').average().data #only low frequency fluctuation
ep_vis_dat=epochsVIS_regr.get_data()
ep_vis_dat_evsub=np.zeros(np.shape(ep_vis_dat))
ch_cnt=1
print('Doing fancy pants evoked removal (visual data).')
for ch in range(np.shape(ep_vis_dat)[1]):
    print('(VIS) Removing evoked from channel '+epochsVIS_regr.ch_names[ch]+  ' ' +str(ch_cnt)+ '/' +str(np.shape(ep_vis_dat)[1]))
    ch_cnt=ch_cnt+1
    cur_ch=ep_vis_dat[:,ch,:]
    cur_ev=ev_filt[ch,]
            
    for tr in range(np.shape(cur_ch)[0]):
        trial=cur_ch[tr,]        
        
        #create regressors
        regr=np.ones([len(cur_ev),2]) #create regressors, intercept and evoked
        regr[:,1]=cur_ev #add evoked
        reg = LinearRegression().fit(regr, trial)
        ev_regr=(regr[:,1]*reg.coef_[1])
        ev_nowarp=(regr[:,1]*reg.coef_[1])
        
        alignment = dtw(ev_regr,trial,step_pattern=symmetricP2,window_type=sakoeChibaWindow,window_args={'window_size': 50} ) #close, seems fastest
        
        #engage
        ev_warp=ev_regr[warp(alignment,index_reference=False)]
        ev_regr[0:len(ev_warp),]=ev_warp
        
        #remove warping artifacts
        ev_regr=mne.filter.filter_data(ev_regr,1000,l_freq=0,h_freq=10,verbose='error')
                
        #after warp, fit may be improved
        regr=np.ones([len(ev_regr),2]) #create regressors, intercept and evoked
        regr[:,1]=ev_regr #add evoked
        reg = LinearRegression().fit(regr, trial)
        ev_regr=(regr[:,1]*reg.coef_[1])                                        
        
        #only apply warping if it leads to lower RMS
        if np.sum(np.sqrt((trial-ev_nowarp)**2))<=np.sum(np.sqrt((trial-ev_regr)**2)):
            ep_vis_dat_evsub[tr,ch,] = trial-ev_nowarp
        else:
            ep_vis_dat_evsub[tr,ch,] = trial-ev_regr               
        
#add cleaned data to epochs structure
epochsVIS_evsub=deepcopy(epochsVIS_regr)
epochsVIS_evsub._data=ep_vis_dat_evsub
epochsVIS_evsub.apply_baseline(baseline=(bl_min,bl_max))  

#Motor - evoked subtraction
ev_filt=epochsMOT_regr.copy().filter(l_freq=0,h_freq=10,verbose='error').average().data #only low frequency fluctuation
ep_mot_dat=epochsMOT_regr.get_data()
ep_mot_dat_evsub=np.zeros(np.shape(ep_mot_dat))
ch_cnt=1
print('Doing fancy pants evoked removal (motor data).')
for ch in range(np.shape(ep_mot_dat)[1]):
    print('(MOT) Removing evoked from channel '+epochsMOT_regr.ch_names[ch_cnt-1]+  ' ' +str(ch_cnt)+ '/' +str(np.shape(ep_mot_dat)[1]))
    ch_cnt=ch_cnt+1
    cur_ch=ep_mot_dat[:,ch,:]
    cur_ev=ev_filt[ch,]
        
    #for every trial..
    for tr in range(np.shape(cur_ch)[0]):
        trial=cur_ch[tr,]        
        
        #create regressors
        regr=np.ones([len(cur_ev),2]) #create regressors, intercept and evoked
        regr[:,1]=cur_ev #add evoked
        reg = LinearRegression().fit(regr, trial)
        ev_regr=(regr[:,1]*reg.coef_[1])
        ev_nowarp=(regr[:,1]*reg.coef_[1])
        
        alignment = dtw(ev_regr,trial,step_pattern=symmetricP2,window_type=sakoeChibaWindow,window_args={'window_size': 50} ) #close, seems fastest
        
        #engage
        ev_warp=ev_regr[warp(alignment,index_reference=False)]
        ev_regr[0:len(ev_warp),]=ev_warp
        
        #remove warping artifacts
        ev_regr=mne.filter.filter_data(ev_regr,1000,l_freq=0,h_freq=10,verbose='error')        
                
        #after warp, fit may be improved
        regr=np.ones([len(ev_regr),2]) #create regressors, intercept and evoked
        regr[:,1]=ev_regr #add evoked
        reg = LinearRegression().fit(regr, trial)
        ev_regr=(regr[:,1]*reg.coef_[1])  
        
        #only apply warping if it leads to lower RMS
        if np.sum(np.sqrt((trial-ev_nowarp)**2))<=np.sum(np.sqrt((trial-ev_regr)**2)):
            ep_mot_dat_evsub[tr,ch,] = trial-ev_nowarp
        else:
            ep_mot_dat_evsub[tr,ch,] = trial-ev_regr

#add cleaned data to epochs structure
epochsMOT_evsub=deepcopy(epochsMOT_regr)
epochsMOT_evsub._data=ep_mot_dat_evsub
epochsMOT_evsub.apply_baseline(baseline=(bl_min,bl_max))  

#before AR, make sure reference channels are not used for threshold determination
if np.any(np.isin(int(subject), [12,14])): #14?
    z_chans=['RC33_z', 'RC13_z', 'RC11_z', 'RC31_z']
else:
    z_chans=['LC33_z', 'LC13_z', 'LC11_z', 'LC31_z']
epochsVIS_evsub.info['bads']=z_chans
epochsMOT_evsub.info['bads']=z_chans


#Autoreject - VIS regressed + evsub
if use_autoreject:
    if len(epochsVIS_evsub)/len(epochsVIS_evsub.drop_log)<0.7:
        print('Pre-cleaning already removed <70% of the data, skipping further rejection')
    else:
        print('VIS (evsub) Using autoreject to discard artifactual epochs')        
        rejectTHRES = ar.get_rejection_threshold(epochsVIS_evsub, decim=2,random_state=42,ch_types='mag') #get AR threshold
        
        drop=epochsVIS_evsub.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
        print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsVIS_evsub.drop_log)*100)+ '%')
        #we want to keep at least 70% of the data; if too stringent, make rejection threshold a bit more liberal
        if len(drop)/len(epochsVIS_evsub.drop_log)<.7: #rejection too harsh
            print('Rejection criterion too conservative, iterating to retain more epochs.')
            while len(drop)/len(epochsVIS_evsub.drop_log)<.7:
                rejectTHRES['mag']=rejectTHRES['mag']*1.01
                drop=epochsVIS_evsub.copy().drop_bad(reject=rejectTHRES,verbose='WARNING')
                print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsVIS_evsub.drop_log)*100)+ '%')    
        
        #apply rejection using the selected threshold
        epochsVIS_evsub=epochsVIS_evsub.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection   

#Autoreject - MOT regressed + evsub
if use_autoreject:
    if len(epochsMOT_evsub)/len(epochsMOT_evsub.drop_log)<0.7:
        print('Pre-cleaning already removed <70% of the data, skipping further rejection')
    else:
        print('MOT (evsub) - Using autoreject to discard artifactual epochs')        
        rejectTHRES = ar.get_rejection_threshold(epochsMOT_evsub, decim=2,random_state=42,ch_types='mag') #get AR threshold
        
        drop=epochsMOT_evsub.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
        print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsMOT_evsub.drop_log)*100)+ '%')
        #we want to keep at least 70% of the data; if too stringent, make rejection threshold a bit more liberal
        if len(drop)/len(epochsMOT_evsub.drop_log)<.7: #rejection too harsh
            print('Rejection criterion too conservative, iterating to retain more epochs.')
            while len(drop)/len(epochsMOT_evsub.drop_log)<.7:
                rejectTHRES['mag']=rejectTHRES['mag']*1.01
                drop=epochsMOT_evsub.copy().drop_bad(reject=rejectTHRES,verbose='WARNING')
                print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsMOT_evsub.drop_log)*100)+ '%')    
        
        #apply rejection using the selected threshold
        epochsMOT_evsub=epochsMOT_evsub.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection     

#Create grand averages
print('Averaging Epochs..')
evokedVIS_evsub=epochsVIS_evsub.average()
evokedMOT_evsub=epochsMOT_evsub.average()

#Save data - keep selected epochs, full epoch dataset & evoked         
print('Saving evoked and epochs..')
epochsVIS_evsub.save(op.join(data_path,'group','Motor','epochs',subject+'_VIS_evsub_OPM-epo.fif'),overwrite=True)
epochsMOT_evsub.save(op.join(data_path,'group','Motor','epochs',subject+'_MOT_evsub_OPM-epo.fif'),overwrite=True)
print('Done saving subject ' +subject)    