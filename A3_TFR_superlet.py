# Here we import the preprocessed epochs and
# compute the TFR for visual onset and reach offset

#Preamble
import os.path as op
from os import listdir
import sys
import mne
import numpy as np

import matplotlib.pyplot as plt 
plt.ion()

#Set up data for import
sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
from superlet import superlet
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'

sub=sys.argv[1]

#Get subject files and sort by type
fifs_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and sub in d]
fifs_epochs.sort()

MEG_VIS_files=[d for d in fifs_epochs if 'MEG' in d and 'VIS' in d and sub in d]
MEG_MOT_files=[d for d in fifs_epochs if 'MEG' in d and 'MOT' in d and sub in d]
OPM_VIS_evsub_files=[d for d in fifs_epochs if 'evsub_OPM' in d and 'VIS' in d and sub in d]
OPM_MOT_evsub_files=[d for d in fifs_epochs if 'evsub_OPM' in d and 'MOT' in d and sub in d]
print('Found following epoch files: ' +MEG_VIS_files[0]+ ', ' +MEG_MOT_files[0]+ ', ' +OPM_VIS_evsub_files[0]+ ', '  +OPM_MOT_evsub_files[0])


#################
## SQUID - MOT ##
#################
 
print('Calculating superlet TFR for SQUID-MEG Visual epochs ')
print('loading data.')
dat=mne.read_epochs(op.join(group_path,'epochs',MEG_VIS_files[0]))

print('selecting and renaming channels.')
#only the MEG channels
dat.pick_types('mag')

#equalize names
if 'ML' in dat.ch_names[0]:
    
    MEG_chan_names_dict = {
        'MLC25-2805': 'meg1',
        'MLF64-2805': 'meg2',
        'MZC02-2805': 'meg3',
        'MLP11-2805': 'meg4'
        }
else:
    MEG_chan_names_dict = {
        'MRP44-2805': 'meg1',
        'MRC16-2805': 'meg2',
        'MRC41-2805': 'meg3',
        'MRP12-2805': 'meg4'     
        }

dat.rename_channels(MEG_chan_names_dict)   

print('extracting data.')
data=dat.get_data().T #time as first dimension

#parameters
samplerate=dat.info['sfreq']
freqs=np.arange(2,100)
order_max=40
order_min=1
c_1=2
print('SQUID-VIS - Calculating superlet: samplerate=' +str(samplerate)+ ', freqs=[' +str(np.min(freqs))+ '-' +str(np.max(freqs))+ '], min order=' +str(order_min)+ ', max order=' +str(order_max)+ ', c1=' +str(c_1)+ ', adaptive=True')
sl = superlet(data,samplerate=samplerate,freqs=freqs,order_max=order_max,order_min=order_min,c_1=c_1,adaptive=True)
print('Done.')

print('creating data structure.')
sl=np.abs(sl)

info=dat.info

#reshape data
ep_dat=np.zeros([np.shape(sl)[3],np.shape(sl)[2],np.shape(sl)[0],np.shape(sl)[1]])
for tr in range(np.shape(sl)[3]):    
    for ch in range(np.shape(sl)[2]):        
        ep_dat[tr,ch,:,:]=sl[:,:,ch,tr]
               
times=dat.times        

#add to epochs structure
#data shape (n_epochs, n_channels, n_freqs, n_times)
TFR_sl_ep = mne.time_frequency.EpochsTFR(
            info,
            ep_dat,
            times, 
            freqs,
            events=dat.events,
            comment="Superlet TF")

print('Saving TFR epochs as TFR_epochs',sub+'_MEG_VIS_sl-tfr.h5')
TFR_sl_ep.save(op.join(group_path, 'TFR_epochs',sub+'_MEG_VIS_sl-tfr.h5'),overwrite=True)
print('Done.')

print('Averaging TFR epochs and saving grand average')
TFR_sl=TFR_sl_ep.average()
TFR_sl.save(op.join(group_path, 'TFR',sub+'_MEG_VIS_sl_ga-tfr.h5'),overwrite=True)
print('Done with SQUID-VIS')


#################
## SQUID - MOT ##
#################

print('Calculating superlet TFR for SQUID-MEG Motor epochs ')
print('loading data.')
dat=mne.read_epochs(op.join(group_path,'epochs',MEG_MOT_files[0]))

print('selecting and renaming channels.')
#only the MEG channels
dat.pick_types('mag')

#equalize names
if 'ML' in dat.ch_names[0]:
    
    MEG_chan_names_dict = {
        'MLC25-2805': 'meg1',
        'MLF64-2805': 'meg2',
        'MZC02-2805': 'meg3',
        'MLP11-2805': 'meg4'
        }
else:
    MEG_chan_names_dict = {
        'MRP44-2805': 'meg1',
        'MRC16-2805': 'meg2',
        'MRC41-2805': 'meg3',
        'MRP12-2805': 'meg4'     
        }

dat.rename_channels(MEG_chan_names_dict)   

print('extracting data.')
data=dat.get_data().T #time as first dimension

#parameters
samplerate=dat.info['sfreq']
freqs=np.arange(2,100)
order_max=40
order_min=1
c_1=2
print('SQUID-MOT - Calculating superlet: samplerate=' +str(samplerate)+ ', freqs=[' +str(np.min(freqs))+ '-' +str(np.max(freqs))+ '], min order=' +str(order_min)+ ', max order=' +str(order_max)+ ', c1=' +str(c_1)+ ', adaptive=True')
sl = superlet(data,samplerate=samplerate,freqs=freqs,order_max=order_max,order_min=order_min,c_1=c_1,adaptive=True)
print('Done.')

print('creating data structure.')
sl=np.abs(sl)

info=dat.info

#reshape data
ep_dat=np.zeros([np.shape(sl)[3],np.shape(sl)[2],np.shape(sl)[0],np.shape(sl)[1]])
for tr in range(np.shape(sl)[3]):    
    for ch in range(np.shape(sl)[2]):        
        ep_dat[tr,ch,:,:]=sl[:,:,ch,tr]
               
times=dat.times        

#add to epochs structure
#data shape (n_epochs, n_channels, n_freqs, n_times)
TFR_sl_ep = mne.time_frequency.EpochsTFR(
            info,
            ep_dat,
            times, 
            freqs,
            events=dat.events,
            comment="Superlet TF")

print('Saving TFR epochs as TFR_epochs',sub+'_MEG_MOT_sl-tfr.h5')
TFR_sl_ep.save(op.join(group_path, 'TFR_epochs',sub+'_MEG_MOT_sl-tfr.h5'),overwrite=True)
print('Done.')

print('Averaging TFR epochs and saving grand average')
TFR_sl=TFR_sl_ep.average()
TFR_sl.save(op.join(group_path, 'TFR',sub+'_MEG_MOT_sl_ga-tfr.h5'),overwrite=True)
print('Done with SQUID-MOT')

###############
## OPM - VIS ##
###############
 
print('Calculating superlet TFR for OPM-MEG Visual epochs ')
print('loading data.')
dat_noev=mne.read_epochs(op.join(group_path,'epochs',OPM_VIS_evsub_files[0]))

print('selecting and renaming channels.')
ref_chans=mne.pick_channels_regexp(dat_noev.info['ch_names'],'^ZC*')
dat_noev.drop_channels(['ZC2_x','ZC2_y','ZC2_z'],on_missing='ignore')

#equalize names
if 'LC' in dat_noev.ch_names[0]:
    chan_names_dict = {
        'LC33_x': 'C33_x',
        'LC33_y': 'C33_y',
        'LC33_z': 'C33_z',
        'LC13_x': 'C13_x',
        'LC13_y': 'C13_y',
        'LC13_z': 'C13_z',
        'LC11_x': 'C11_x',
        'LC11_y': 'C11_y',
        'LC11_z': 'C11_z',
        'LC31_x': 'C31_x',
        'LC31_y': 'C31_y',
        'LC31_z': 'C31_z'
        }       
else:
    chan_names_dict = { 
        'RC33_x': 'C33_x',
        'RC33_y': 'C33_y',
        'RC33_z': 'C33_z',
        'RC13_x': 'C13_x',
        'RC13_y': 'C13_y',
        'RC13_z': 'C13_z',
        'RC11_x': 'C11_x',
        'RC11_y': 'C11_y',
        'RC11_z': 'C11_z',
        'RC31_x': 'C31_x',
        'RC31_y': 'C31_y',
        'RC31_z': 'C31_z',
        }

dat_noev.rename_channels(chan_names_dict)               

print('extracting data.')
data_noev=dat_noev.get_data().T

#parameters
samplerate=dat_noev.info['sfreq']
freqs=np.arange(2,100)
order_max=40
order_min=1
c_1=2
print('OPM-VIS noEV - Calculating superlet: samplerate=' +str(samplerate)+ ', freqs=[' +str(np.min(freqs))+ '-' +str(np.max(freqs))+ '], min order=' +str(order_min)+ ', max order=' +str(order_max)+ ', c1=' +str(c_1)+ ', adaptive=True')
sl_noev = superlet(data_noev,samplerate=samplerate,freqs=freqs,order_max=order_max,order_min=order_min,c_1=c_1,adaptive=True)
print('Done.')

print('creating data structure.')
sl_noev=np.abs(sl_noev)

info=dat_noev.info

#reshape data        
ep_dat_noev=np.zeros([np.shape(sl_noev)[3],np.shape(sl_noev)[2],np.shape(sl_noev)[0],np.shape(sl_noev)[1]])
for tr in range(np.shape(sl_noev)[3]):    
    for ch in range(np.shape(sl_noev)[2]):        
        ep_dat_noev[tr,ch,:,:]=sl_noev[:,:,ch,tr]        
        
times=dat_noev.times        

#add to epochs structure
#data shape (n_epochs, n_channels, n_freqs, n_times)
TFR_sl_noev_ep = mne.time_frequency.EpochsTFR(
            info,
            ep_dat_noev,
            times, 
            freqs,
            events=dat_noev.events,
            comment="Superlet TF (ev sub)")

print('Saving evoked subtracted TFR epochs as TFR_epochs',sub+'_OPM_VIS_sl_noev-tfr.h5')
TFR_sl_noev_ep.save(op.join(group_path, 'TFR_epochs',sub+'_OPM_VIS_sl_noev-tfr.h5'),overwrite=True)
print('Done.')

print('Averaging TFR epochs and saving grand average')
TFR_sl_noev=TFR_sl_noev_ep.average()
TFR_sl_noev.save(op.join(group_path, 'TFR',sub+'_OPM_VIS_sl_noev_ga-tfr.h5'),overwrite=True)
print('Done with OPM-VIS')

###############
## OPM - MOT ##
###############
 
print('Calculating superlet TFR for OPM-MEG Motor epochs ')
print('loading data.')
dat_noev=mne.read_epochs(op.join(group_path,'epochs',OPM_MOT_evsub_files[0]))

print('selecting and renaming channels.')
ref_chans=mne.pick_channels_regexp(dat_noev.info['ch_names'],'^ZC*')
dat_noev.drop_channels(['ZC2_x','ZC2_y','ZC2_z'],on_missing='ignore')

#equalize names
if 'LC' in dat_noev.ch_names[0]:
    chan_names_dict = {
        'LC33_x': 'C33_x',
        'LC33_y': 'C33_y',
        'LC33_z': 'C33_z',
        'LC13_x': 'C13_x',
        'LC13_y': 'C13_y',
        'LC13_z': 'C13_z',
        'LC11_x': 'C11_x',
        'LC11_y': 'C11_y',
        'LC11_z': 'C11_z',
        'LC31_x': 'C31_x',
        'LC31_y': 'C31_y',
        'LC31_z': 'C31_z'
        }       
else:
    chan_names_dict = { 
        'RC33_x': 'C33_x',
        'RC33_y': 'C33_y',
        'RC33_z': 'C33_z',
        'RC13_x': 'C13_x',
        'RC13_y': 'C13_y',
        'RC13_z': 'C13_z',
        'RC11_x': 'C11_x',
        'RC11_y': 'C11_y',
        'RC11_z': 'C11_z',
        'RC31_x': 'C31_x',
        'RC31_y': 'C31_y',
        'RC31_z': 'C31_z',
        }

dat_noev.rename_channels(chan_names_dict)               

print('extracting data.')
data_noev=dat_noev.get_data().T

#parameters
samplerate=dat_noev.info['sfreq']
freqs=np.arange(2,100)
order_max=40
order_min=1
c_1=2
print('OPM-MOT noEV - Calculating superlet: samplerate=' +str(samplerate)+ ', freqs=[' +str(np.min(freqs))+ '-' +str(np.max(freqs))+ '], min order=' +str(order_min)+ ', max order=' +str(order_max)+ ', c1=' +str(c_1)+ ', adaptive=True')
sl_noev = superlet(data_noev,samplerate=samplerate,freqs=freqs,order_max=order_max,order_min=order_min,c_1=c_1,adaptive=True)
print('Done.')

print('creating data structure.')
sl_noev=np.abs(sl_noev)
info=dat_noev.info

#reshape data        
ep_dat_noev=np.zeros([np.shape(sl_noev)[3],np.shape(sl_noev)[2],np.shape(sl_noev)[0],np.shape(sl_noev)[1]])
for tr in range(np.shape(sl_noev)[3]):    
    for ch in range(np.shape(sl_noev)[2]):        
        ep_dat_noev[tr,ch,:,:]=sl_noev[:,:,ch,tr]        
        
times=dat_noev.times        

#add to epochs structure
#data shape (n_epochs, n_channels, n_freqs, n_times)
TFR_sl_noev_ep = mne.time_frequency.EpochsTFR(
            info,
            ep_dat_noev,
            times, 
            freqs,
            events=dat_noev.events,
            comment="Superlet TF (ev sub)")

print('Saving evoked subtracted TFR epochs as TFR_epochs',sub+'_OPM_MOT_sl_noev-tfr.h5')
TFR_sl_noev_ep.save(op.join(group_path, 'TFR_epochs',sub+'_OPM_MOT_sl_noev-tfr.h5'),overwrite=True)
print('Done.')

print('Averaging TFR epochs and saving grand average')
TFR_sl_noev=TFR_sl_noev_ep.average()
TFR_sl_noev.save(op.join(group_path, 'TFR',sub+'_OPM_MOT_sl_noev_ga-tfr.h5'),overwrite=True)
print('Done with OPM-MOT')  