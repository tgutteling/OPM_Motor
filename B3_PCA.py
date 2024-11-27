import pickle
import os.path as op
import numpy as np
from os import listdir
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import pickle

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'

print('Collecting data files')
#Get subject files and sort by type
VIS_SQUID_bursts = [d for d in listdir(op.join(group_path,'bursts')) if 'SQUID_VIS' in d]
VIS_SQUID_bursts.sort()
VIS_OPM_bursts = [d for d in listdir(op.join(group_path,'bursts')) if 'OPM_VIS' in d]
VIS_OPM_bursts.sort()

print('Gathering Beta burst waveforms')
print('SQUID..')
SQUID_waveforms = []
for vis_file in VIS_SQUID_bursts:
    file = open(op.join(group_path,'bursts',vis_file), 'rb')
    bs = pickle.load(file)    
    for i in range(len(bs)):
        am_bs = len(bs[i]["waveform"])
        bs_samp_ix = np.random.choice(np.arange(am_bs), int(am_bs))
        bs_samp = np.vstack(bs[i]['waveform'])[bs_samp_ix, :]
        SQUID_waveforms.append(bs_samp)        
    
SQUID_waveforms = np.vstack(SQUID_waveforms)

#Remove outer 10 percentiles
print('Doing some cleaning')
waveforms_medians = np.median(SQUID_waveforms, axis=1)
clean_ixs = np.where((waveforms_medians > np.percentile(waveforms_medians, 10)) & (waveforms_medians < np.percentile(waveforms_medians, 90)))[0]
SQUID_waveforms_clean = SQUID_waveforms[clean_ixs]

print('Saving selected waveforms')
wf_path = op.join(group_path, "SQUID_waveforms_clean.pkl")
pickle.dump(SQUID_waveforms_clean, open(wf_path, "wb"))

#scale before PCA
scaler = RobustScaler().fit(SQUID_waveforms_clean)
SQUID_waveforms_scaled = scaler.transform(SQUID_waveforms_clean)

print('OPM..')
OPM_waveforms = {'x': [], 'y':[], 'z': []}
for vis_file in VIS_OPM_bursts:
    file = open(op.join(group_path,'bursts',vis_file), 'rb')
    bs = pickle.load(file)
    for key in bs.keys():
        cur=bs[key]
        for i in range(len(cur)):
            am_bs = len(cur[i]["waveform"])        
            bs_samp_ix = np.random.choice(np.arange(am_bs), int(am_bs*1))
            bs_samp = np.vstack(cur[i]['waveform'])[bs_samp_ix, :]
            OPM_waveforms[key].append(bs_samp)        
    
#Remove outer 10 percentiles
print('Doing some cleaning')
OPM_waveforms_clean = {'x': [], 'y':[], 'z': []}
OPM_waveforms_scaled = {'x': [], 'y':[], 'z': []}
OPM_waveforms_medians = {'x': [], 'y':[], 'z': []}
for k in OPM_waveforms.keys():    
    OPM_waveforms[k] = np.vstack(OPM_waveforms[k])
    OPM_waveforms_medians[k] = np.median(OPM_waveforms[k], axis=1)
    clean_ixs = np.where((OPM_waveforms_medians[k] > np.percentile(OPM_waveforms_medians[k], 10)) & (OPM_waveforms_medians[k] < np.percentile(OPM_waveforms_medians[k], 90)))[0]
    OPM_waveforms_clean[k] = OPM_waveforms[k][clean_ixs]
    #scale before PCA
    scaler = RobustScaler().fit(OPM_waveforms_clean[k])
    OPM_waveforms_scaled[k] = scaler.transform(OPM_waveforms_clean[k])


print('Saving selected waveforms')
wf_path = op.join(group_path, "OPM_waveforms_clean.pkl")
pickle.dump(OPM_waveforms_clean, open(wf_path, "wb"))

#combine all bursts and fit common PCA
burst_all=np.vstack([SQUID_waveforms_scaled,OPM_waveforms_scaled['x'], OPM_waveforms_scaled['y']]) #all but Z

#Create combined PCA
comb_pca_allnoZ = PCA(n_components=20)
comb_pca_allnoZ=comb_pca_allnoZ.fit(burst_all)    

#save PCA
print('Saving PCA')
PCA_path = op.join(group_path, "comb_PCA_allnoZ.pkl")
pickle.dump(comb_pca_allnoZ, open(PCA_path, "wb"))
print('Done')