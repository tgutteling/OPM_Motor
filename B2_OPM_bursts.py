#Here we take the preprocessed -OPM- data and extract some beta bursts.

#Preamble
import os.path as op
from os import listdir
import sys
import mne
import numpy as np
from fooof import FOOOF
import pickle

import matplotlib.pyplot as plt 
plt.ion()

sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg/Motor/')
from burst_detection import extract_bursts

#for burst detection we need the following:
# 1 raw_trials: raw data for trial (trialxtime)
# 2 tf: time-frequency decomposition for each trial (trial x freq x time)
# 3 times: time steps
# 4 search_freqs: frequency limits to search within for bursts (should be wider than band_lims)
# 5 band_lims: keep bursts whose peak frequency falls within these limits
# 6 aperiodic_spectrum: aperiodic spectrum
# 7 sfreq: sampling rate
# 8 w_size: window size to extract burst waveforms (leave to default)

#parameters to set

#5 - band_lims
beta_lims = [15, 30] #note that this is different from the original analysis and done because of the noise contamination of the OPM sensors <15Hz
fit_range=[15,45] #range for PSD fitting; TFR epochs is also constricted to this range

#1 - get raw trials /3 times
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'
fifs_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and 'evsub_OPM' in d]
fifs_epochs.sort()

#Argument input
subject=sys.argv[1]     
        
#get subect files        
OPM_VIS_file=[d for d in fifs_epochs if 'VIS' in d and subject in d]
OPM_MOT_file=[d for d in fifs_epochs if 'MOT' in d and subject in d]

print('Loading OPM-VIS data')
dat_VIS=mne.read_epochs(op.join(group_path,'epochs',OPM_VIS_file[0]))    

# 7 - sampling freq
sfreq=dat_VIS.info['sfreq']

#get times
times_VIS=dat_VIS.times

#only the X and Y channels
picks_x=mne.pick_channels_regexp(dat_VIS.info['ch_names'],'^L.*x')+mne.pick_channels_regexp(dat_VIS.info['ch_names'],'^R.*x')
picks_y=mne.pick_channels_regexp(dat_VIS.info['ch_names'],'^L.*y')+mne.pick_channels_regexp(dat_VIS.info['ch_names'],'^R.*y')
picks_z=mne.pick_channels_regexp(dat_VIS.info['ch_names'],'^L.*z')+mne.pick_channels_regexp(dat_VIS.info['ch_names'],'^R.*z')
raw_trials_VIS_x=[]
raw_trials_VIS_y=[]
raw_trials_VIS_z=[]
for s in picks_x:    
    raw_trials_VIS_x.append(dat_VIS.get_data()[:,s,:])
for s in picks_y:    
    raw_trials_VIS_y.append(dat_VIS.get_data()[:,s,:])
for s in picks_z:    
    raw_trials_VIS_z.append(dat_VIS.get_data()[:,s,:])

print('Loading OPM-MOT data')    
dat_MOT=mne.read_epochs(op.join(group_path,'epochs',OPM_MOT_file[0]))    

#get times
times_MOT=dat_MOT.times

#only the X and Y directions
picks_x=mne.pick_channels_regexp(dat_MOT.info['ch_names'],'^L.*x')+mne.pick_channels_regexp(dat_MOT.info['ch_names'],'^R.*x')
picks_y=mne.pick_channels_regexp(dat_MOT.info['ch_names'],'^L.*y')+mne.pick_channels_regexp(dat_MOT.info['ch_names'],'^R.*y')
picks_z=mne.pick_channels_regexp(dat_MOT.info['ch_names'],'^L.*z')+mne.pick_channels_regexp(dat_MOT.info['ch_names'],'^R.*z')
raw_trials_MOT_x=[]
raw_trials_MOT_y=[]
raw_trials_MOT_z=[]
for s in picks_x:    
    raw_trials_MOT_x.append(dat_MOT.get_data()[:,s,:])
for s in picks_y:    
    raw_trials_MOT_y.append(dat_MOT.get_data()[:,s,:])
for s in picks_z:    
    raw_trials_MOT_z.append(dat_MOT.get_data()[:,s,:])

#2 - TFR
print('Getting TFRs')
TFR_epochs_VIS = [d for d in listdir(op.join(group_path,'TFR_epochs')) if '.h5' in d and 'OPM' in d and subject in d and 'sl_noev-tfr' in d and 'VIS' in d]
TFR_epochs_MOT = [d for d in listdir(op.join(group_path,'TFR_epochs')) if '.h5' in d and 'OPM' in d and subject in d and 'sl_noev-tfr' in d and 'MOT' in d]

#load data
dat_TFR_VIS=mne.time_frequency.read_tfrs(op.join(group_path,'TFR_epochs',TFR_epochs_VIS[0]))    
dat_TFR_VIS[0].crop(fmin=fit_range[0],fmax=fit_range[1]) #crop to limited freq range 
freqs=dat_TFR_VIS[0].freqs
search_range= np.where((freqs >= 15) & (freqs <= 33))[0]

#reorder
picks_x=mne.pick_channels_regexp(dat_TFR_VIS[0].info['ch_names'],'^.*x')
picks_y=mne.pick_channels_regexp(dat_TFR_VIS[0].info['ch_names'],'^.*y')
picks_z=mne.pick_channels_regexp(dat_TFR_VIS[0].info['ch_names'],'^.*z')
tf_trials_VIS_x=[]
tf_trials_VIS_y=[]
tf_trials_VIS_z=[]
psd_VIS_x=[]
psd_VIS_y=[]
psd_VIS_z=[]
for s in picks_x:
    tf_trials_VIS_x.append(dat_TFR_VIS[0].data[:,s,search_range,:])
    psd_VIS_x.append(np.mean(np.mean(dat_TFR_VIS[0].data[:,s,:,:],axis=0),axis=1))
for s in picks_y:
    tf_trials_VIS_y.append(dat_TFR_VIS[0].data[:,s,search_range,:])
    psd_VIS_y.append(np.mean(np.mean(dat_TFR_VIS[0].data[:,s,:,:],axis=0),axis=1))    
for s in picks_z:
    tf_trials_VIS_z.append(dat_TFR_VIS[0].data[:,s,search_range,:])
    psd_VIS_z.append(np.mean(np.mean(dat_TFR_VIS[0].data[:,s,:,:],axis=0),axis=1))    
    
#load data
dat_TFR_MOT=mne.time_frequency.read_tfrs(op.join(group_path,'TFR_epochs',TFR_epochs_MOT[0]))        
dat_TFR_MOT[0].crop(fmin=fit_range[0],fmax=fit_range[1]) #crop to limited freq range 

#reorder
picks_x=mne.pick_channels_regexp(dat_TFR_MOT[0].info['ch_names'],'^.*x')
picks_y=mne.pick_channels_regexp(dat_TFR_MOT[0].info['ch_names'],'^.*y')
picks_z=mne.pick_channels_regexp(dat_TFR_MOT[0].info['ch_names'],'^.*z')
tf_trials_MOT_x=[]
tf_trials_MOT_y=[]
tf_trials_MOT_z=[]
psd_MOT_x=[]
psd_MOT_y=[]
psd_MOT_z=[]
for s in picks_x:
    tf_trials_MOT_x.append(dat_TFR_MOT[0].data[:,s,search_range,:])   
    psd_MOT_x.append(np.mean(np.mean(dat_TFR_MOT[0].data[:,s,:,:],axis=0),axis=1))
for s in picks_y:
    tf_trials_MOT_y.append(dat_TFR_MOT[0].data[:,s,search_range,:])   
    psd_MOT_y.append(np.mean(np.mean(dat_TFR_MOT[0].data[:,s,:,:],axis=0),axis=1))
for s in picks_z:
    tf_trials_MOT_z.append(dat_TFR_MOT[0].data[:,s,search_range,:])   
    psd_MOT_z.append(np.mean(np.mean(dat_TFR_MOT[0].data[:,s,:,:],axis=0),axis=1))

#4 - search freqs
search_freqs=freqs[search_range]

#6 - aperiodic spectrum
print('Calculating spectrum')
PSD_VIS_ap_x=[]
for ch in range(np.shape(psd_VIS_x)[0]):
    ff_vis = FOOOF()
    ff_vis.fit(freqs, psd_VIS_x[ch], [fit_range[0],fit_range[1]])
    ap_fit_v = 10 ** ff_vis._ap_fit
    PSD_VIS_ap_x.append(ap_fit_v[search_range].reshape(-1, 1))

PSD_VIS_ap_y=[]
for ch in range(np.shape(psd_VIS_y)[0]):
    ff_vis = FOOOF()
    ff_vis.fit(freqs, psd_VIS_y[ch], [fit_range[0],fit_range[1]])
    ap_fit_v = 10 ** ff_vis._ap_fit
    PSD_VIS_ap_y.append(ap_fit_v[search_range].reshape(-1, 1))
    
PSD_VIS_ap_z=[]
for ch in range(np.shape(psd_VIS_z)[0]):
    ff_vis = FOOOF()
    ff_vis.fit(freqs, psd_VIS_z[ch], [fit_range[0],fit_range[1]])
    ap_fit_v = 10 ** ff_vis._ap_fit
    PSD_VIS_ap_z.append(ap_fit_v[search_range].reshape(-1, 1))    
    
PSD_MOT_ap_x=[]
for ch in range(np.shape(psd_MOT_x)[0]):
    ff_mot = FOOOF()
    ff_mot.fit(freqs, psd_MOT_x[ch], [fit_range[0],fit_range[1]])
    ap_fit_m = 10 ** ff_mot._ap_fit
    PSD_MOT_ap_x.append(ap_fit_m[search_range].reshape(-1, 1))    
    
PSD_MOT_ap_y=[]
for ch in range(np.shape(psd_MOT_y)[0]):
    ff_mot = FOOOF()
    ff_mot.fit(freqs, psd_MOT_y[ch], [fit_range[0],fit_range[1]])
    ap_fit_m = 10 ** ff_mot._ap_fit
    PSD_MOT_ap_y.append(ap_fit_m[search_range].reshape(-1, 1))    
    
PSD_MOT_ap_z=[]
for ch in range(np.shape(psd_MOT_z)[0]):
    ff_mot = FOOOF()
    ff_mot.fit(freqs, psd_MOT_z[ch], [fit_range[0],fit_range[1]])
    ap_fit_m = 10 ** ff_mot._ap_fit
    PSD_MOT_ap_z.append(ap_fit_m[search_range].reshape(-1, 1))        
    
#Now extract some bursties
VIS_bursts_x=[]
for ch in range(np.shape(raw_trials_VIS_x)[0]): 
    print('VIS: X axis, Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_VIS_x)[0]))
    chan_vis_burst_x = extract_bursts(
            raw_trials_VIS_x[ch], 
            tf_trials_VIS_x[ch],
            times_VIS, 
            freqs[search_range], 
            beta_lims, 
            PSD_VIS_ap_x[ch], 
            sfreq,            
            w_size=0.26
        )
    VIS_bursts_x.append(chan_vis_burst_x)

VIS_bursts_y=[]
for ch in range(np.shape(raw_trials_VIS_y)[0]): 
    print('VIS: Y axis, Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_VIS_y)[0]))
    chan_vis_burst_y = extract_bursts(
            raw_trials_VIS_y[ch], 
            tf_trials_VIS_y[ch],
            times_VIS, 
            freqs[search_range], 
            beta_lims, 
            PSD_VIS_ap_y[ch], 
            sfreq,            
            w_size=0.26
        )
    VIS_bursts_y.append(chan_vis_burst_y)

VIS_bursts_z=[]
for ch in range(np.shape(raw_trials_VIS_z)[0]): 
    print('VIS: Z axis, Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_VIS_z)[0]))
    chan_vis_burst_z = extract_bursts(
            raw_trials_VIS_z[ch], 
            tf_trials_VIS_z[ch],
            times_VIS, 
            freqs[search_range], 
            beta_lims, 
            PSD_VIS_ap_z[ch], 
            sfreq,            
            w_size=0.26
        )
    VIS_bursts_z.append(chan_vis_burst_z)
        
VIS_bursts = dict()
VIS_bursts['x'] = VIS_bursts_x
VIS_bursts['y'] = VIS_bursts_y
VIS_bursts['z'] = VIS_bursts_z    

#save 
print('Saving..')
file=op.join(group_path,'bursts',subject+ '_OPM_VIS.pkl')
pickle.dump(VIS_bursts,open(file,'wb'))
print('Done.')
        
MOT_bursts_x=[]
for ch in range(np.shape(raw_trials_MOT_x)[0]): 
    print('MOT: X axis, Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_MOT_x)[0]))
    chan_mot_burst_x = extract_bursts(
            raw_trials_MOT_x[ch], 
            tf_trials_MOT_x[ch],
            times_MOT, 
            freqs[search_range], 
            beta_lims, 
            PSD_MOT_ap_x[ch], 
            sfreq,            
            w_size=0.26
        )
    MOT_bursts_x.append(chan_mot_burst_x)    
    
MOT_bursts_y=[]
for ch in range(np.shape(raw_trials_MOT_y)[0]): 
    print('MOT: Y axis, Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_MOT_y)[0]))
    chan_mot_burst_y = extract_bursts(
            raw_trials_MOT_y[ch], 
            tf_trials_MOT_y[ch],
            times_MOT, 
            freqs[search_range], 
            beta_lims, 
            PSD_MOT_ap_y[ch], 
            sfreq,            
            w_size=0.26
        )
    MOT_bursts_y.append(chan_mot_burst_y)    
    
MOT_bursts_z=[]
for ch in range(np.shape(raw_trials_MOT_z)[0]): 
    print('MOT: Z axis, Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_MOT_z)[0]))
    chan_mot_burst_z = extract_bursts(
            raw_trials_MOT_z[ch], 
            tf_trials_MOT_z[ch],
            times_MOT, 
            freqs[search_range], 
            beta_lims, 
            PSD_MOT_ap_z[ch], 
            sfreq,            
            w_size=0.26
        )
    MOT_bursts_z.append(chan_mot_burst_z)        
        
MOT_bursts = dict()
MOT_bursts['x'] = MOT_bursts_x
MOT_bursts['y'] = MOT_bursts_y
MOT_bursts['z'] = MOT_bursts_z  

#save 
print('Saving..')
file=op.join(group_path,'bursts',subject+ '_OPM_MOT.pkl')
pickle.dump(MOT_bursts,open(file,'wb'))
print('Done.')

    