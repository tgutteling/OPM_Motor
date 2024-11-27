# Here we import the raw MEG Squid data and
# > remove break periods
# > remove spikes
# > run ICA
# > Check and reject components based on visual inspection (continuous and epoched)
# > Remove artifact components
# > filter cleaned data
# > epoch cleaned data
# > downsample to 1Khz
# > Calculate evoked field

#Preamble
#from logging import warning
import os.path as op
from os import listdir
import mne
import numpy as np
import autoreject as ar
from meegkit import dss

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
    if op.exists(op.join(data_path, str(dirs[d]).zfill(2), 'MEG-'+str(dirs[d]).zfill(2)+'-joystick-01.ds')) or op.exists(op.join(data_path, str(dirs[d]).zfill(2), 'MEG-'+str(dirs[d]).zfill(2)+'-joystick-11.ds')):
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
    
#Retrieve sessions associated with this subject
print('Retreiving sessions..') 
ds_files=[d for d in listdir(op.join(data_path, subject)) if '.ds' in d and  'MEG' in d and 'joystick' in d]

data_files=[]
ses_nr=[]
for d in range(len(ds_files)):
    data_files.append(ds_files[d])
    ses_nr.append(int(ds_files[d][-5:-3]))
         
ses_nr=np.sort(ses_nr)
runs=[]                              
for s in range(len(ses_nr)):
    runs.append(data_files[0][:-5] +str(ses_nr[s]).zfill(2) +'.ds')                    

#Task parameters 
#Triggers (names): (guessed)
    #'13'     1 - start/end block
    #'end'    8 - end block
    #'2'      2 - fixation [1s - 2s]
    #'3'      3 - visual stim [2s]
    #'4'      4 - delay [0.45s - 1s]
    #'5'      5 - target [0-1s]
    #'6'      6 - feedback [1.5-4s]
    #'7'      7 - feedback-motor [2-2.9s]

    
tminVIS,tmaxVIS = -1, 2 #time before and after trigger
tminMOT,tmaxMOT = -1, 1.5 #time before and after trigger
bl_min,bl_max=-0.1,0 #baseline start and end (None for either first or last sample)
highpass,lowpass = 1,100 #bandpass frequencies
line_freq=[50] #notch filter frequency(/ies)
final_samp_rate=1000 #Output sampling rate
use_previous_rejection=0 #load rejection parameters from previous dataset, for consistency when reanalysing with different parameters
use_autoreject=1

all_raw=[]
for i_run in runs:
    print('Loading ' +i_run)
    file_name=op.join(data_path, subject, i_run) 
    
    raw = mne.io.read_raw_ctf(file_name, preload=True)    

    #apply third order gradient compensation (if not already done)
    raw.apply_gradient_compensation(3)
    
    #get events
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
    
    #Remove spikes from data
    #Reject by amplitude, use standard deviation of the raw data
    raw_tmp=raw.copy().resample(500).filter(l_freq=1, h_freq=100)    
    dat=raw_tmp.get_data(picks='meg',reject_by_annotation='omit')    
    n_sds=5 #how many SD to set as threshold
    spike_annots, spike_bads = mne.preprocessing.annotate_amplitude(
        raw_tmp,
        peak=np.median(dat)+np.std(dat)*n_sds
        )        
    raw.set_annotations(raw.annotations + spike_annots) #add these annotations to the raw data
    
    all_raw.append(raw) #add annotated raw session to the whole
    
raw=mne.concatenate_raws(all_raw, on_mismatch='warn')    

#get events from concatenated RAW
events, event_dict=mne.events_from_annotations(raw)
        
if not np.any(np.isin(int(subject), [15])):
    #split data into MEG and EEG (EOG)
    picks_eog=['EEG063-2800','EEG064-2800']
    raw_meg=raw.copy().pick_types(meg=True,eeg=False,ref_meg=True)
    raw_eog=raw.copy().pick_channels(picks_eog)
    #save annotations
    annots=raw_meg.annotations
   
    #Remove line noise - since this method does not use filtering, we can apply it before ICA
    #We have to chunk the data to make it manageable, use 50s chunks
    print('Removing line noise from MEG sensors, this make take a while..')
    smp=np.shape(raw_meg.get_data())[1]
    chunk=int(smp/(np.ceil((smp/1200)/50)))
    zapped,_=dss.dss_line(raw_meg.get_data().T,line_freq[0],raw_meg.info['sfreq'],nremove=3,blocksize=chunk)
    print('Done.')
    
    print('Removing line noise from EOG, this will be faster.')
    smp=np.shape(raw_eog.get_data())[1]
    chunk=int(smp/(np.ceil((smp/1200)/50)))
    zapped_eog,_=dss.dss_line(raw_eog.get_data().T,line_freq[0],raw_meg.info['sfreq'],nremove=1,blocksize=chunk)
    print('Done.')

    #create new raw structure
    raw_lnrem=mne.io.RawArray(zapped.T,raw_meg.info)
    raw_eog_lnrem=mne.io.RawArray(zapped_eog.T,raw_eog.info)
    #combine back into a single structure
    raw_lnrem.add_channels([raw_eog_lnrem])
    raw_lnrem._annotations=annots
else:
    print('Removing line noise from MEG sensors, this make take a while..')
    smp=np.shape(raw.get_data())[1]
    chunk=int(smp/(np.ceil((smp/1200)/50)))
    zapped,_=dss.dss_line(raw.get_data().T,line_freq[0],raw.info['sfreq'],nremove=3,blocksize=chunk)
    print('Done.')
    raw_lnrem=mne.io.RawArray(zapped.T,raw.info)
    raw_lnrem._annotations=raw.annotations

#Now that we have collected all session and removed all gross artefacts from the data, we can use the data for ICA
print('Creating downsampled, filtered RAW data for ICA')
raw_ica=raw_lnrem.copy().resample(500).filter(l_freq=1, h_freq=100)    

#Compute ICA
print('Computing ICA')
ica = mne.preprocessing.ICA(n_components=30,method='fastica',random_state=42)
ica.fit(raw_ica)

        
if not use_previous_rejection:
    #Visually inspect components
    ica.plot_sources(raw_ica)
    
    #for comparison, use epoched data    
    raw_ep=raw_lnrem.copy().filter(l_freq=1, h_freq=100)
    event_id_VIS={'3' : event_dict['3']}                  
    tmp_epochs=mne.Epochs(raw_ep,events,event_id=event_id_VIS,tmin=tminVIS,tmax=tmaxVIS,preload=True)
    tmp_epochs.resample(500)
    ica.plot_sources(tmp_epochs)
    
#Mark BAD ICA components 
ICA_reject = {
    '01' : [0,1,2,3,4,5],
    '02' : [0,1,2,3],
    '04' : [0,1,2,18,27],
    '06' : [0,1,22,28,29],
    '07' : [0,2,3,5,6,7,8,9,12,12,15,22,28,29], #depends on manual cleaning of continuous data
    '08' : [0,1,2,3,9,23],
    '09' : [0,1,2,7,17,18,22],
    '10' : [0,1,2,3,4,5,6,23,29],
    '12' : [0,1,2],
    '14' : [0,1,2,3,9],
    '15' : [0,1,2,3,4,6],
    '16' : [0,2,4,5,29],
    '17' : [0,1,2,4],
    '18' : [0,1,2,3,4,5,6,8],
    '19' : [0,1,2,5],
    '21' : [0,1,2,5,6],
    '23' : [0,1,6,9,18],
    '24' : [0,1,2]
    }

#Apply ICA to raw, unfiltered data
if use_previous_rejection:
    raw_ic_clean=ica.apply(raw_lnrem,exclude=ICA_reject[subject])    
else:
    raw_ic_clean=ica.apply(raw_lnrem)       

#To reduce futher computational costs, reduce the dataset to the relevant sensors only
#The closest four SQUID sensors should be:
    #MLC25 - MLF64 - MZC02 - MLP11 (left sensors)
    #MRP44 - MRC16 - MRC41 - MRP12 (right sensors)

picks_left=['MLC25-2805','MLF64-2805','MZC02-2805','MLP11-2805']
picks_right=['MRP44-2805','MRC16-2805','MRC41-2805','MRP12-2805']
picks_eog=['EEG063-2800','EEG064-2800']

#Check with OPM data whether to pick left or right sensors
if np.any(np.isin(int(subject), [12,14])):
    raw_ic_clean.pick_channels(picks_right+picks_eog)
else:
    raw_ic_clean.pick_channels(picks_left+picks_eog)

#Bandpass filter
print('Wide Bandpass filter')
raw_filt=raw_ic_clean.copy().filter(l_freq=highpass, h_freq=lowpass,n_jobs=-1) # bandpass

if np.any(np.isin(int(subject), [15])):
    raw_filt.resample(1200)


#Define VISual and MOTor event triggers
event_id_VIS={'3' : event_dict['3']}                 
event_id_MOT={'7' : event_dict['7']}     
events_VIS, event_dict_VIS=mne.events_from_annotations(raw_filt, event_id=event_id_VIS)
events_MOT, event_dict_MOT=mne.events_from_annotations(raw_filt, event_id=event_id_MOT)

if not np.any(np.isin(int(subject), [15])):
    #Use EOG to regress out ocular artefacts - Estimate betas; we'll base the regression model on the visual stimulation
    epochsTMP=mne.Epochs(raw_filt,events_VIS,event_id=event_id_VIS,tmin=tminVIS,tmax=tmaxVIS,preload=True).subtract_evoked() #create epochs for regression estimation
    picks_eog=['EEG063-2800','EEG064-2800']
    epochsTMP.set_eeg_reference(ref_channels=[])
    
    #estimate the EOG model
    model_EOG = mne.preprocessing.EOGRegression(picks='meg',picks_artifact=picks_eog).fit(epochsTMP) #new version

#Create epochs
epochsVIS=mne.Epochs(raw_filt,events_VIS,event_id=event_id_VIS,tmin=tminVIS,tmax=tmaxVIS,preload=True) #create epoch object for AR thresholding
epochsMOT=mne.Epochs(raw_filt,events_MOT,event_id=event_id_MOT,tmin=tminMOT,tmax=tmaxMOT,preload=True) #create epoch object for AR thresholding

if not np.any(np.isin(int(subject), [15])):
    #apply EOG regression to epochs
    epochsMOT.set_eeg_reference(ref_channels=[])
    epochsVIS.set_eeg_reference(ref_channels=[])
    epochsVIS=model_EOG.apply(epochsVIS)
    epochsMOT=model_EOG.apply(epochsMOT)

#After regression, epochs need to be baselined again
epochsVIS.apply_baseline(baseline=(bl_min,bl_max))   
epochsMOT.apply_baseline(baseline=(bl_min,bl_max))   

#now resample to output sampling rate
epochsVIS.resample(final_samp_rate)
epochsMOT.resample(final_samp_rate)

#Autoreject - VIS
if use_autoreject:
    if len(epochsVIS)/len(epochsVIS.drop_log)<0.7:
        print('Pre-cleaning already removed <70% of the data, skipping further rejection')
    else:
        print('Using autoreject to discard artifactual epochs')        
        rejectTHRES = ar.get_rejection_threshold(epochsVIS, decim=2,random_state=42,ch_types='mag') #get AR threshold
        
        drop=epochsVIS.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
        print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsVIS.drop_log)*100)+ '%')
        #we want to keep at least 70% of the data; if too stringent, make rejection threshold a bit more liberal
        if len(drop)/len(epochsVIS.drop_log)<.7: #rejection too harsh
            print('Rejection criterion too conservative, iterating to retain more epochs.')
            while len(drop)/len(epochsVIS.drop_log)<.7:
                rejectTHRES['mag']=rejectTHRES['mag']*1.01
                drop=epochsVIS.copy().drop_bad(reject=rejectTHRES,verbose='WARNING')
                print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsVIS.drop_log)*100)+ '%')
        
        #apply rejection using the selected threshold
        epochsVIS=epochsVIS.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
                
#Autoreject - MOT
if use_autoreject:
    if len(epochsVIS)/len(epochsMOT.drop_log)<0.7:
        print('Pre-cleaning already removed <70% of the data, skipping further rejection')
    else:
        print('Using autoreject to discard artifactual epochs')        
        rejectTHRES = ar.get_rejection_threshold(epochsMOT, decim=2,random_state=42,ch_types='mag') #get AR threshold
        
        drop=epochsMOT.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
        print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsMOT.drop_log)*100)+ '%')
        #we want to keep at least 70% of the data; if too stringent, make rejection threshold a bit more liberal
        if len(drop)/len(epochsMOT.drop_log)<.7: #rejection too harsh
            print('Rejection criterion too conservative, iterating to retain more epochs.')
            while len(drop)/len(epochsMOT.drop_log)<.7:
                rejectTHRES['mag']=rejectTHRES['mag']*1.01
                drop=epochsMOT.copy().drop_bad(reject=rejectTHRES,verbose='WARNING')
                print('Threshold: ' +str(rejectTHRES['mag'])+ ' - Portion of data kept: ' +str(len(drop)/len(epochsMOT.drop_log)*100)+ '%')    
        
        #apply rejection using the selected threshold
        epochsMOT=epochsMOT.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection

#Create grand averages
print('Averaging Epochs..')
evokedVIS=epochsVIS.average()
evokedMOT=epochsMOT.average()

#Save data - keep selected epochs, full epoch dataset & evoked         
print('Saving evoked and epochs..')
epochsVIS.save(op.join(data_path,'group','Motor','epochs',subject+'_VIS_MEG-epo.fif'),overwrite=True)
epochsMOT.save(op.join(data_path,'group','Motor','epochs',subject+'_MOT_MEG-epo.fif'),overwrite=True)
evokedVIS.save(op.join(data_path,'group','Motor','evoked',subject+'_VIS_MEG-ave.fif'),overwrite=True)
evokedMOT.save(op.join(data_path,'group','Motor','evoked',subject+'_MOT_MEG-ave.fif'),overwrite=True)
print('Done.')              