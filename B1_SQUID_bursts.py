#Here we take the preprocessed -SQUID- data and extract some beta bursts.

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
beta_lims = [13, 30]

#1 - get raw trials /3 times
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'
fifs_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and 'MEG' in d]
fifs_epochs.sort()

#Argument input
subject=sys.argv[1]  
        
#get subect files        
MEG_VIS_file=[d for d in fifs_epochs if 'MEG' in d and 'VIS' in d and subject in d]
MEG_MOT_file=[d for d in fifs_epochs if 'MEG' in d and 'MOT' in d and subject in d]

print('Loading SQUID-VIS data')
dat_VIS=mne.read_epochs(op.join(group_path,'epochs',MEG_VIS_file[0]))    

# 7 - sampling freq
sfreq=dat_VIS.info['sfreq']

#get times
times_VIS=dat_VIS.times

#only the MEG channels
dat_VIS.pick_types('mag')
raw_trials_VIS=[]
for s in range(np.shape(dat_VIS.get_data())[1]):
    raw_trials_VIS.append(dat_VIS.get_data()[:,s,:])

print('Loading SQUID-MOT data')    
dat_MOT=mne.read_epochs(op.join(group_path,'epochs',MEG_MOT_file[0]))    

#get times
times_MOT=dat_MOT.times

#only the MEG channels
dat_MOT.pick_types('mag')
raw_trials_MOT=[]
for s in range(np.shape(dat_MOT.get_data())[1]):
    raw_trials_MOT.append(dat_MOT.get_data()[:,s,:])    

#2 - TFR
print('Getting TFRs')
TFR_epochs_VIS = [d for d in listdir(op.join(group_path,'TFR_epochs')) if '.h5' in d and 'MEG' in d and subject in d and 'sl-tfr' in d and 'VIS' in d]
TFR_epochs_MOT = [d for d in listdir(op.join(group_path,'TFR_epochs')) if '.h5' in d and 'MEG' in d and subject in d and 'sl-tfr' in d and 'MOT' in d]

#load data
dat_TFR_VIS=mne.time_frequency.read_tfrs(op.join(group_path,'TFR_epochs',TFR_epochs_VIS[0]))        
freqs=dat_TFR_VIS[0].freqs
search_range= np.where((freqs >= 10) & (freqs <= 33))[0]

#reorder
tf_trials_VIS=[]
psd_VIS=[]
for s in range(np.shape(dat_TFR_VIS[0].data)[1]):
    tf_trials_VIS.append(dat_TFR_VIS[0].data[:,s,search_range,:])
    psd_VIS.append(np.mean(np.mean(dat_TFR_VIS[0].data[:,s,:,:],axis=0),axis=1))
    
#load data
dat_TFR_MOT=mne.time_frequency.read_tfrs(op.join(group_path,'TFR_epochs',TFR_epochs_MOT[0]))        

#reorder
tf_trials_MOT=[]
psd_MOT=[]
for s in range(np.shape(dat_TFR_MOT[0].data)[1]):
    tf_trials_MOT.append(dat_TFR_MOT[0].data[:,s,search_range,:])   
    psd_MOT.append(np.mean(np.mean(dat_TFR_MOT[0].data[:,s,:,:],axis=0),axis=1))

#4 - search freqs
search_freqs=freqs[search_range]

#6 - aperiodic spectrum
print('Calculating spectrum')
PSD_VIS_ap=[]
for ch in range(np.shape(psd_VIS)[0]):
    ff_vis = FOOOF()
    ff_vis.fit(freqs, psd_VIS[ch], [2,99])
    ap_fit_v = 10 ** ff_vis._ap_fit
    PSD_VIS_ap.append(ap_fit_v[search_range].reshape(-1, 1))
    
PSD_MOT_ap=[]
for ch in range(np.shape(psd_MOT)[0]):
    ff_mot = FOOOF()
    ff_mot.fit(freqs, psd_MOT[ch], [2,99])
    ap_fit_m = 10 ** ff_mot._ap_fit
    PSD_MOT_ap.append(ap_fit_m[search_range].reshape(-1, 1))    
    
#Now extract some bursties
VIS_bursts=[]
for ch in range(np.shape(raw_trials_VIS)[0]): 
    print('VIS: Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_VIS)[0]))
    chan_vis_burst = extract_bursts(
            raw_trials_VIS[ch], 
            tf_trials_VIS[ch],
            times_VIS, 
            freqs[search_range], 
            beta_lims, 
            PSD_VIS_ap[ch], 
            sfreq,            
            w_size=0.26
        )
    VIS_bursts.append(chan_vis_burst)
    
#save 
print('Saving..')
file=op.join(group_path,'bursts',subject+ '_SQUID_VIS.pkl')
pickle.dump(VIS_bursts,open(file,'wb'))
print('Done.')
        
MOT_bursts=[]
for ch in range(np.shape(raw_trials_MOT)[0]): 
    print('MOT: Extracting bursts for channel '  +str(ch+1)+ '/' +str(np.shape(raw_trials_MOT)[0]))
    chan_mot_burst = extract_bursts(
            raw_trials_MOT[ch], 
            tf_trials_MOT[ch],
            times_MOT, 
            freqs[search_range], 
            beta_lims, 
            PSD_MOT_ap[ch], 
            sfreq,            
            w_size=0.26
        )
    MOT_bursts.append(chan_mot_burst)    
    
#save 
print('Saving..')
file=op.join(group_path,'bursts',subject+ '_SQUID_MOT.pkl')
pickle.dump(MOT_bursts,open(file,'wb'))
print('Done.')

    