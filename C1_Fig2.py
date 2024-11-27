import pickle
import os.path as op
import numpy as np
from os import listdir
import mne
import matplotlib.pyplot as plt
import scipy 
from mne.stats import permutation_cluster_1samp_test
from mne.stats import spatio_temporal_cluster_1samp_test
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'


#SQUID
VIS_SQUID_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'MEG_VIS_sl_ga' in d]
MOT_SQUID_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'MEG_MOT_sl_ga' in d]
VIS_SQUID_TFRs.sort()
MOT_SQUID_TFRs.sort()

print('SQUID: Loading Visual TFR data')
SQUID_vis_TFR=[]
for s in VIS_SQUID_TFRs:
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))    
    SQUID_vis_TFR.append(tmp)
        
print('SQUID: Loading Motor TFR data')
SQUID_mot_TFR=[]
for s in MOT_SQUID_TFRs:
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    SQUID_mot_TFR.append(tmp)

SQUID_vis_avTFR=mne.grand_average(SQUID_vis_TFR)
SQUID_mot_avTFR=mne.grand_average(SQUID_mot_TFR)

#baseline MOT using VIS
bl_time=[-0.5,0]
bl_ix=np.where((SQUID_vis_avTFR.times>=bl_time[0]) & (SQUID_vis_avTFR.times<=bl_time[1]))
bl=np.mean(SQUID_vis_avTFR.data[:,:,bl_ix],axis=3)
SQUID_mot_avTFR.data=(SQUID_mot_avTFR.data-bl)/bl

#do the same for the individual subjects
for s in range(len(SQUID_mot_TFR)):
    bl=np.mean(SQUID_vis_TFR[s].data[:,:,bl_ix],axis=3)
    SQUID_mot_TFR[s].data=(SQUID_mot_TFR[s].data-bl)


#OPM
VIS_OPM_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'OPM_VIS_sl_noev' in d]
MOT_OPM_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'OPM_MOT_sl_noev' in d]
VIS_OPM_TFRs.sort()
MOT_OPM_TFRs.sort()

print('OPM: Loading Visual TFR data')
OPM_vis_TFR=dict({'x':[], 'y':[], 'z':[]})
for s in VIS_OPM_TFRs:
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    tmp.info['bads']=[]
    picks_x=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*x')
    picks_y=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*y')
    picks_z=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*z')
    OPM_vis_TFR['x'].append(tmp.copy().pick(picks=picks_x))
    OPM_vis_TFR['y'].append(tmp.copy().pick(picks=picks_y))
    OPM_vis_TFR['z'].append(tmp.copy().pick(picks=picks_z))
    
OPM_vis_avTFR=dict({'x':[], 'y':[], 'z':[]})    
for k in OPM_vis_TFR.keys():
    OPM_vis_avTFR[k]=mne.grand_average(OPM_vis_TFR[k])    

print('OPM: Loading Motor TFR data')
OPM_mot_TFR=dict({'x':[], 'y':[], 'z':[]})
for s in MOT_OPM_TFRs:
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    tmp.info['bads']=[]
    picks_x=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*x')
    picks_y=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*y')
    picks_z=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*z')
    OPM_mot_TFR['x'].append(tmp.copy().pick(picks=picks_x))
    OPM_mot_TFR['y'].append(tmp.copy().pick(picks=picks_y))
    OPM_mot_TFR['z'].append(tmp.copy().pick(picks=picks_z))
    
OPM_mot_avTFR=dict({'x':[], 'y':[], 'z':[]})    
for k in OPM_mot_TFR.keys():
    OPM_mot_avTFR[k]=mne.grand_average(OPM_mot_TFR[k])    

bl_time=[-0.5,0]
bl_ix=np.where((OPM_vis_avTFR['x'].times>=bl_time[0]) & (OPM_vis_avTFR['x'].times<=bl_time[1]))
for axs in OPM_mot_avTFR.keys():
    bl=np.mean(OPM_vis_avTFR[axs].data[:,:,bl_ix],axis=3)
    OPM_mot_avTFR[axs].data=(OPM_mot_avTFR[axs].data-bl)/bl

for s in range(len(OPM_mot_TFR['y'])):
    bl=np.mean(OPM_vis_TFR['y'][s].data[:,:,bl_ix],axis=3)
    OPM_mot_TFR['y'][s].data=(OPM_mot_TFR['y'][s].data-bl)
for s in range(len(OPM_mot_TFR['x'])):
    bl=np.mean(OPM_vis_TFR['x'][s].data[:,:,bl_ix],axis=3)
    OPM_mot_TFR['x'][s].data=(OPM_mot_TFR['x'][s].data-bl)    

#stats
#collapse data into single matrix    
#subs x freqs x times
#baseline application is 'contrast' to test (one sample)
#let's first crop to max 1.75s post stim to avoid edge artefact
for s in range(len(SQUID_vis_TFR)):
    SQUID_vis_TFR[s].crop(tmin=-1,tmax=1.75)
for s in range(len(SQUID_mot_TFR)):
    SQUID_mot_TFR[s].crop(tmin=-.75,tmax=1.25)    
for s in range(len(OPM_vis_TFR['y'])):
    OPM_vis_TFR['y'][s].crop(tmin=-1,tmax=1.75)    
for s in range(len(OPM_vis_TFR['x'])):
    OPM_vis_TFR['x'][s].crop(tmin=-1,tmax=1.75)
for s in range(len(OPM_mot_TFR['y'])):
    OPM_mot_TFR['y'][s].crop(tmin=-.75,tmax=1.25)        
for s in range(len(OPM_mot_TFR['x'])):
    OPM_mot_TFR['x'][s].crop(tmin=-.75,tmax=1.25)            


all_SQUID_data=np.zeros((len(SQUID_vis_TFR),np.shape(SQUID_vis_TFR[0].data)[1],np.shape(SQUID_vis_TFR[0].copy().crop(tmin=0).data)[2]))
all_SQUID_plot=np.zeros((len(SQUID_vis_TFR),np.shape(SQUID_vis_TFR[0].data)[1],np.shape(SQUID_vis_TFR[0].data)[2]))
for s in range(len(SQUID_vis_TFR)):    #for all subs    
    tmp_plot=SQUID_vis_TFR[s].copy().apply_baseline(baseline=(-0.5, 0),mode='percent')
    SQUID_vis_TFR[s].apply_baseline(baseline=(-0.5, 0),mode='mean')    
    tmp=SQUID_vis_TFR[s].copy().crop(tmin=0)    
    all_SQUID_data[s,:,:]=np.average(tmp.data,axis=0)
    all_SQUID_plot[s,:,:]=np.average(tmp_plot.data,axis=0)
    
mot_SQUID_data=np.zeros((len(SQUID_mot_TFR),np.shape(SQUID_mot_TFR[0].data)[1],np.shape(SQUID_mot_TFR[0].data)[2]))
for s in range(len(SQUID_mot_TFR)):    #for all subs    
    mot_SQUID_data[s,:,:]=np.average(SQUID_mot_TFR[s].data,axis=0)    
    
#same for OPM  
all_opm_y_data=np.zeros((len(OPM_vis_TFR['y']),np.shape(OPM_vis_TFR['y'][0].data)[1],np.shape(OPM_vis_TFR['y'][0].copy().crop(tmin=0).data)[2]))
all_opm_x_data=np.zeros((len(OPM_vis_TFR['x']),np.shape(OPM_vis_TFR['x'][0].data)[1],np.shape(OPM_vis_TFR['x'][0].copy().crop(tmin=0).data)[2]))
all_opm_y_plot=np.zeros((len(OPM_vis_TFR['y']),np.shape(OPM_vis_TFR['y'][0].data)[1],np.shape(OPM_vis_TFR['y'][0].data)[2]))
all_opm_x_plot=np.zeros((len(OPM_vis_TFR['x']),np.shape(OPM_vis_TFR['x'][0].data)[1],np.shape(OPM_vis_TFR['y'][0].data)[2]))
for s in range(len(OPM_vis_TFR['y'])):
    tmp_plot=OPM_vis_TFR['y'][s].copy().apply_baseline(baseline=(-0.5, 0),mode='percent')
    OPM_vis_TFR['y'][s].apply_baseline(baseline=(-0.5, 0),mode='mean')
    tmp=OPM_vis_TFR['y'][s].copy().crop(tmin=0)
    all_opm_y_data[s,:,:]=np.average(tmp.data,axis=0)    
    all_opm_y_plot[s,:,:]=np.average(tmp_plot.data,axis=0)
    tmp_plot=OPM_vis_TFR['x'][s].copy().apply_baseline(baseline=(-0.5, 0),mode='percent')
    OPM_vis_TFR['x'][s].apply_baseline(baseline=(-0.5, 0),mode='mean')
    tmp=OPM_vis_TFR['x'][s].copy().crop(tmin=0)
    all_opm_x_data[s,:,:]=np.average(tmp.data,axis=0)
    all_opm_x_plot[s,:,:]=np.average(tmp_plot.data,axis=0)    
    
all_opm_y_mot=np.zeros((len(OPM_mot_TFR['y']),np.shape(OPM_mot_TFR['y'][0].data)[1],np.shape(OPM_mot_TFR['y'][0].data)[2]))
all_opm_x_mot=np.zeros((len(OPM_mot_TFR['x']),np.shape(OPM_mot_TFR['x'][0].data)[1],np.shape(OPM_mot_TFR['y'][0].data)[2]))
for s in range(len(OPM_mot_TFR['y'])):
    all_opm_y_mot[s,:,:]=np.average(OPM_mot_TFR['y'][s].data,axis=0)
    all_opm_x_mot[s,:,:]=np.average(OPM_mot_TFR['x'][s].data,axis=0)


#As this is a one sample t-test, contrasting the baseline with activation, we should only consider clusters in the activation interval    
#compute adjacency
freqs_SQUID=SQUID_vis_TFR[0].freqs    
times_SQUID=SQUID_vis_TFR[0].copy().crop(tmin=0).times
adjacency_SQUID = mne.stats.combine_adjacency(len(freqs_SQUID), len(times_SQUID))

freqs_motSQUID=SQUID_mot_TFR[0].freqs    
times_motSQUID=SQUID_mot_TFR[0].times
adjacency_motSQUID = mne.stats.combine_adjacency(len(freqs_motSQUID), len(times_motSQUID))

freqs_opm=OPM_vis_TFR['y'][0].freqs    
times_opm=OPM_vis_TFR['y'][0].copy().crop(tmin=0).times
adjacency_opm = mne.stats.combine_adjacency(len(freqs_opm), len(times_opm))

freqs_motopm=OPM_mot_TFR['y'][0].freqs    
times_motopm=OPM_mot_TFR['y'][0].times
adjacency_motopm = mne.stats.combine_adjacency(len(freqs_motopm), len(times_motopm))

#set switch to either calculate or load
load_stats=1 #load stats or recalculate?

if load_stats:
    print('Loading stats')
    stats_file = open(op.join(group_path, "TFR_stats.pkl"),'rb')
    stats=pickle.load(stats_file)
else:

    #create dict to save stats
    stat_out=['T_obs', 'clusters', 'cluster_p_values']
    stats={mod: {cond: {out: [] for out in stat_out} for cond in ['vis','mot']} for mod in ['SQUID','OPMY','OPMX']}
    
    #SQUID-VIS
    p=0.05
    n_permutations=1000
    df=np.shape(all_SQUID_data)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_SQUID, clusters_SQUID, cluster_p_values_SQUID, H0_SQUID = permutation_cluster_1samp_test(
                                            all_SQUID_data, 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_SQUID,
                                            n_jobs=-1,
                                            out_type='mask',
                                            verbose=True)
    
    stats['SQUID']['vis']['T_obs']=T_obs_SQUID
    stats['SQUID']['vis']['cluster']=clusters_SQUID
    stats['SQUID']['vis']['cluster_p_values']=cluster_p_values_SQUID
    
    df=np.shape(mot_SQUID_data)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_motSQUID, clusters_motSQUID, cluster_p_values_motSQUID, H0_motSQUID = permutation_cluster_1samp_test(
                                            mot_SQUID_data, 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_motSQUID,
                                            n_jobs=-1,
                                            out_type='mask',
                                            verbose=True)
    
    stats['SQUID']['mot']['T_obs']=T_obs_motSQUID
    stats['SQUID']['mot']['cluster']=clusters_motSQUID
    stats['SQUID']['mot']['cluster_p_values']=cluster_p_values_motSQUID
    
    
    #OPM Y
    df=np.shape(all_opm_y_data)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_opm_y, clusters_opm_y, cluster_p_values_opm_y, H0_opm_y = permutation_cluster_1samp_test(
                                            all_opm_y_data, 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_opm,
                                            n_jobs=-1,
                                            out_type='mask',
                                            verbose=True)
    
    stats['OPMY']['vis']['T_obs']=T_obs_opm_y
    stats['OPMY']['vis']['cluster']=clusters_opm_y
    stats['OPMY']['vis']['cluster_p_values']=cluster_p_values_opm_y
    
    df=np.shape(all_opm_y_mot)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_motopm_y, clusters_motopm_y, cluster_p_values_motopm_y, H0_opm_y = permutation_cluster_1samp_test(
                                            all_opm_y_mot, 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_motopm,
                                            n_jobs=-1,
                                            out_type='mask',
                                            verbose=True)
    
    stats['OPMY']['mot']['T_obs']=T_obs_motopm_y
    stats['OPMY']['mot']['cluster']=clusters_motopm_y
    stats['OPMY']['mot']['cluster_p_values']=cluster_p_values_motopm_y
    
    #OPM X
    df=np.shape(all_opm_x_data)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_opm_x, clusters_opm_x, cluster_p_values_opm_x, H0_opm_x = permutation_cluster_1samp_test(
                                            all_opm_x_data, 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_opm,
                                            n_jobs=-1,
                                            out_type='mask',
                                            verbose=True)
    
    
    stats['OPMX']['vis']['T_obs']=T_obs_opm_x
    stats['OPMX']['vis']['cluster']=clusters_opm_x
    stats['OPMX']['vis']['cluster_p_values']=cluster_p_values_opm_x
    
    
    df=np.shape(all_opm_x_mot)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_motopm_x, clusters_motopm_x, cluster_p_values_motopm_x, H0_opm_x = permutation_cluster_1samp_test(
                                            all_opm_x_mot, 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_motopm,
                                            n_jobs=-1,
                                            out_type='mask',
                                            verbose=True)
    
    stats['OPMX']['mot']['T_obs']=T_obs_motopm_x
    stats['OPMX']['mot']['cluster']=clusters_motopm_x
    stats['OPMX']['mot']['cluster_p_values']=cluster_p_values_motopm_x
    
    #Save stats
    print('Saving stats')       
    stats_path = op.join(group_path, "TFR_stats.pkl")
    pickle.dump(stats, open(stats_path, "wb"))
    print('All done.')   

T_obs_SQUID = stats['SQUID']['vis']['T_obs']
clusters_SQUID = stats['SQUID']['vis']['cluster']
cluster_p_values_SQUID = stats['SQUID']['vis']['cluster_p_values'] 
T_obs_motSQUID = stats['SQUID']['mot']['T_obs']
clusters_motSQUID = stats['SQUID']['mot']['cluster']
cluster_p_values_motSQUID = stats['SQUID']['mot']['cluster_p_values'] 
T_obs_opm_y = stats['OPMY']['vis']['T_obs']
clusters_opm_y = stats['OPMY']['vis']['cluster']
cluster_p_values_opm_y = stats['OPMY']['vis']['cluster_p_values']
T_obs_motopm_y = stats['OPMY']['mot']['T_obs']
clusters_motopm_y = stats['OPMY']['mot']['cluster']
cluster_p_values_motopm_y = stats['OPMY']['mot']['cluster_p_values']
T_obs_opm_x = stats['OPMX']['vis']['T_obs']
clusters_opm_x = stats['OPMX']['vis']['cluster']
cluster_p_values_opm_x = stats['OPMX']['vis']['cluster_p_values']
T_obs_motopm_x = stats['OPMX']['mot']['T_obs']
clusters_motopm_x = stats['OPMX']['mot']['cluster']
cluster_p_values_motopm_x = stats['OPMX']['mot']['cluster_p_values']


#plot results
ga_SQUID=mne.grand_average(SQUID_vis_TFR)
ga_opm_y=mne.grand_average(OPM_vis_TFR['y'])
ga_opm_x=mne.grand_average(OPM_vis_TFR['x'])

#Create time-frequency matrix of significant clusters (t-values)
T_obs_plot_SQUID = np.nan * np.ones_like(T_obs_SQUID)
for c, p_val in zip(clusters_SQUID, cluster_p_values_SQUID):
    if p_val <= 0.05:
        T_obs_plot_SQUID[c] = T_obs_SQUID[c]
        
T_obs_plot_motSQUID = np.nan * np.ones_like(T_obs_motSQUID)
for c, p_val in zip(clusters_motSQUID, cluster_p_values_motSQUID):
    if p_val <= 0.05:
        T_obs_plot_motSQUID[c] = T_obs_motSQUID[c]        
        
T_obs_plot_opm_y = np.nan * np.ones_like(T_obs_opm_y)
for c, p_val in zip(clusters_opm_y, cluster_p_values_opm_y):
    if p_val <= 0.05:
        T_obs_plot_opm_y[c] = T_obs_opm_y[c]        
        
T_obs_plot_motopm_y = np.nan * np.ones_like(T_obs_motopm_y)
for c, p_val in zip(clusters_motopm_y, cluster_p_values_motopm_y):
    if p_val <= 0.05:
        T_obs_plot_motopm_y[c] = T_obs_motopm_y[c]                

T_obs_plot_opm_x = np.nan * np.ones_like(T_obs_opm_x)
for c, p_val in zip(clusters_opm_x, cluster_p_values_opm_x):
    if p_val <= 0.05:
        T_obs_plot_opm_x[c] = T_obs_opm_x[c]        

T_obs_plot_motopm_x = np.nan * np.ones_like(T_obs_motopm_x)
for c, p_val in zip(clusters_motopm_x, cluster_p_values_motopm_x):
    if p_val <= 0.05:
        T_obs_plot_motopm_x[c] = T_obs_motopm_x[c]                        


#for plotting, create a dilated mask to indicate significance
bool_mask=scipy.ndimage.binary_dilation(np.invert(np.isnan(T_obs_plot_SQUID)))
SQUID_outline=np.ones(np.shape(bool_mask))*np.nan
SQUID_outline[bool_mask]=1
bool_mask=scipy.ndimage.binary_dilation(np.invert(np.isnan(T_obs_plot_motSQUID)))
motSQUID_outline=np.ones(np.shape(bool_mask))*np.nan
motSQUID_outline[bool_mask]=1
bool_mask=scipy.ndimage.binary_dilation(np.invert(np.isnan(T_obs_plot_opm_y)))
opmY_outline=np.ones(np.shape(bool_mask))*np.nan
opmY_outline[bool_mask]=1
bool_mask=scipy.ndimage.binary_dilation(np.invert(np.isnan(T_obs_plot_motopm_y)))
motopmY_outline=np.ones(np.shape(bool_mask))*np.nan
motopmY_outline[bool_mask]=1
bool_mask=scipy.ndimage.binary_dilation(np.invert(np.isnan(T_obs_plot_opm_x)))
opmX_outline=np.ones(np.shape(bool_mask))*np.nan
opmX_outline[bool_mask]=1
bool_mask=scipy.ndimage.binary_dilation(np.invert(np.isnan(T_obs_plot_motopm_x)))
motopmX_outline=np.ones(np.shape(bool_mask))*np.nan
motopmX_outline[bool_mask]=1
    


#####################
# Plot power values #
#####################

SQUID_av=np.average(all_SQUID_plot,axis=0)
opmY_av=np.average(all_opm_y_plot,axis=0)
opmX_av=np.average(all_opm_x_plot,axis=0)

t1=SQUID_vis_TFR[0].time_as_index(-0.5)[0]
t2=SQUID_vis_TFR[0].time_as_index(1.75)[0]
SQUID_av=SQUID_av[:,t1:]
times=SQUID_vis_TFR[0].times[t1:]

#pad outline to size with baseline
pad=np.ones([np.shape(SQUID_outline)[0],np.shape(SQUID_av)[1]-np.shape(SQUID_outline)[1]])*np.nan
outline_pad=np.concatenate((pad,SQUID_outline),axis=1)

#create power-fill region
T_obs_plot_SQUID[np.invert(np.isnan(T_obs_plot_SQUID))]=1
power_pad=np.concatenate((pad,T_obs_plot_SQUID),axis=1)
power_fill_SQUID=power_pad*SQUID_av

SQUID_mot_avTFR.crop(tmin=-.75,tmax=1.25)
T_obs_plot_motSQUID[np.invert(np.isnan(T_obs_plot_motSQUID))]=1
power_fill_motSQUID=T_obs_plot_motSQUID*np.mean(SQUID_mot_avTFR.data,axis=0)

#OPM
t1=OPM_vis_TFR['y'][0].time_as_index(-0.5)[0]
t2=OPM_vis_TFR['y'][0].time_as_index(1.75)[0]
opmY_av=opmY_av[:,t1:]
opmX_av=opmX_av[:,t1:]

#pad outline to size with baseline
pad=np.ones([np.shape(opmY_outline)[0],np.shape(opmY_av)[1]-np.shape(opmY_outline)[1]])*np.nan
outline_padY=np.concatenate((pad,opmY_outline),axis=1)
outline_padX=np.concatenate((pad,opmX_outline),axis=1)

#create power-fill region
T_obs_plot_opm_y[np.invert(np.isnan(T_obs_plot_opm_y))]=1
T_obs_plot_opm_x[np.invert(np.isnan(T_obs_plot_opm_x))]=1
power_padY=np.concatenate((pad,T_obs_plot_opm_y),axis=1)
power_padX=np.concatenate((pad,T_obs_plot_opm_x),axis=1)
power_fill_opmY=power_padY*opmY_av
power_fill_opmX=power_padX*opmX_av

OPM_mot_avTFR['y'].crop(tmin=-.75,tmax=1.25)
T_obs_plot_motopm_y[np.invert(np.isnan(T_obs_plot_motopm_y))]=1
power_fill_motOPMY=T_obs_plot_motopm_y*np.mean(OPM_mot_avTFR['y'].data,axis=0)

OPM_mot_avTFR['x'].crop(tmin=-.75,tmax=1.25)
T_obs_plot_motopm_x[np.invert(np.isnan(T_obs_plot_motopm_x))]=1
power_fill_motOPMX=T_obs_plot_motopm_x*np.mean(OPM_mot_avTFR['x'].data,axis=0)

fig,axs=plt.subplots(3,2)
global_ylim=[8,80]
global_xlim=[-.5,1.6]
global_mot_xlim=[-.6,1.2]
SQUID_scale=0.2
pos=axs[0,0].imshow(SQUID_av, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], SQUID_vis_TFR[0].freqs[0], SQUID_vis_TFR[0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-SQUID_scale, vmax=SQUID_scale)
axs[0,0].imshow(outline_pad, cmap=plt.cm.gray,extent=[times[0],times[-1], freqs_SQUID[0], freqs_SQUID[-1]],
           aspect='auto', origin='lower')
axs[0,0].imshow(power_fill_SQUID, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], SQUID_vis_TFR[0].freqs[0], SQUID_vis_TFR[0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-SQUID_scale, vmax=SQUID_scale)

lims=axs[0,0].get_ylim()
axs[0,0].vlines(0,lims[0],lims[1],colors='k',linestyles='dashed')
axs[0,0].set_ylim(global_ylim)
axs[0,0].set_xlim(global_xlim)
axs[0,0].set_title('SQUID-MEG')                
axs[0,0].set_ylabel('Frequency (Hz)')

pos=axs[0,1].imshow(np.mean(SQUID_mot_avTFR.data,axis=0), cmap=plt.cm.RdBu_r,
           extent=[times_motSQUID[0],times_motSQUID[-1], SQUID_mot_TFR[0].freqs[0], SQUID_mot_TFR[0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-SQUID_scale, vmax=SQUID_scale)
axs[0,1].imshow(motSQUID_outline, cmap=plt.cm.gray,extent=[times_motSQUID[0],times_motSQUID[-1], freqs_motSQUID[0], freqs_motSQUID[-1]],
           aspect='auto', origin='lower')
axs[0,1].imshow(power_fill_motSQUID, cmap=plt.cm.RdBu_r,
           extent=[times_motSQUID[0], times_motSQUID[-1], SQUID_mot_TFR[0].freqs[0], SQUID_mot_TFR[0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-SQUID_scale, vmax=SQUID_scale)

lims=axs[0,1].get_ylim()
axs[0,1].vlines(0,lims[0],lims[1],colors='k',linestyles='dashed')
fig.colorbar(pos,ax=axs[0,1])
axs[0,1].set_ylim(global_ylim)
axs[0,1].set_xlim(global_mot_xlim)
axs[0,1].set_title('SQUID-MEG')                
axs[0,1].set_ylabel('Frequency (Hz)')


#OPM Y
opm_scale=0.2
pos=axs[1,0].imshow(opmY_av, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1],OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)
axs[1,0].imshow(outline_padY, cmap=plt.cm.gray,extent=[times[0],times[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower')
axs[1,0].imshow(power_fill_opmY, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)

lims=axs[1,0].get_ylim()
axs[1,0].vlines(0,lims[0],lims[1],colors='k',linestyles='dashed')
axs[1,0].set_ylim(global_ylim)
axs[1,0].set_xlim(global_xlim)
axs[1,0].set_title('4He-OPM$_{radial}$')                
axs[1,0].set_ylabel('Frequency (Hz)')

pos=axs[1,1].imshow(np.mean(OPM_mot_avTFR['y'].data,axis=0), cmap=plt.cm.RdBu_r,
           extent=[times_motopm[0], times_motopm[-1],OPM_mot_TFR['y'][0].freqs[0], OPM_mot_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)
axs[1,1].imshow(motopmY_outline, cmap=plt.cm.gray,extent=[times_motopm[0],times_motopm[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower')
axs[1,1].imshow(power_fill_motOPMY, cmap=plt.cm.RdBu_r,
           extent=[times_motopm[0], times_motopm[-1], OPM_mot_TFR['y'][0].freqs[0], OPM_mot_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)

lims=axs[1,1].get_ylim()
axs[1,1].vlines(0,lims[0],lims[1],colors='k',linestyles='dashed')
fig.colorbar(pos,ax=axs[1,1])
axs[1,1].set_ylim(global_ylim)
axs[1,1].set_xlim(global_mot_xlim)
axs[1,1].set_title('4He-OPM$_{radial}$')                
axs[1,1].set_ylabel('Frequency (Hz)')


#OPM X
pos=axs[2,0].imshow(opmX_av, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], OPM_vis_TFR['x'][0].freqs[0],OPM_vis_TFR['x'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)
axs[2,0].imshow(outline_padX, cmap=plt.cm.gray,extent=[times[0],times[-1], OPM_vis_TFR['x'][0].freqs[0], OPM_vis_TFR['x'][0].freqs[-1]],
           aspect='auto', origin='lower')
axs[2,0].imshow(power_fill_opmX, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], OPM_vis_TFR['x'][0].freqs[0], OPM_vis_TFR['x'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)

lims=axs[2,0].get_ylim()
axs[2,0].vlines(0,lims[0],lims[1],colors='k',linestyles='dashed')
axs[2,0].set_ylim(global_ylim)
axs[2,0].set_xlim(global_xlim)
axs[2,0].set_title('4He-OPM$_{tangential}$')                
axs[2,0].set_xlabel('Time (s)')
axs[2,0].set_ylabel('Frequency (Hz)')

pos=axs[2,1].imshow(np.mean(OPM_mot_avTFR['x'].data,axis=0), cmap=plt.cm.RdBu_r,
           extent=[times_motopm[0], times_motopm[-1],OPM_mot_TFR['y'][0].freqs[0], OPM_mot_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)
axs[2,1].imshow(motopmX_outline, cmap=plt.cm.gray,extent=[times_motopm[0],times_motopm[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower')
axs[2,1].imshow(power_fill_motOPMX, cmap=plt.cm.RdBu_r,
           extent=[times_motopm[0], times_motopm[-1], OPM_mot_TFR['y'][0].freqs[0], OPM_mot_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower', vmin=-opm_scale, vmax=opm_scale)

lims=axs[2,1].get_ylim()
axs[2,1].vlines(0,lims[0],lims[1],colors='k',linestyles='dashed')
axs[2,1].set_ylim(global_ylim)
axs[2,1].set_xlim(global_mot_xlim)
fig.colorbar(pos,ax=axs[2,1])
axs[2,1].set_title('4He-OPM$_{radial}$')                
axs[2,1].set_xlabel('Time (s)')
axs[2,1].set_ylabel('Frequency (Hz)')

fig.set_figheight(10)
fig.set_figwidth(11)
fig.savefig(op.join(group_path,'Fig2A_Group_TFR_stats_Power.pdf'))   
fig.savefig(op.join(group_path,'Fig2A_Group_TFR_stats_Power.png'))   


#outline only
fig,axs=plt.subplots(3,2)
global_ylim=[8,80]
global_xlim=[-.5,1.6]
global_mot_xlim=[-.6,1.2]
SQUID_scale=0.2
power_fill_SQUID[~np.isnan(power_fill_SQUID)]=0
axs[0,0].imshow(outline_pad, cmap=plt.cm.gray,extent=[times[0],times[-1], freqs_SQUID[0], freqs_SQUID[-1]],
           aspect='auto', origin='lower')

axs[0,0].set_ylim(global_ylim)
axs[0,0].set_xlim(global_xlim)
axs[0,0].set_title('SQUID-MEG')                
axs[0,0].set_ylabel('Frequency (Hz)')

power_fill_motSQUID[~np.isnan(power_fill_motSQUID)]=0
axs[0,1].imshow(motSQUID_outline, cmap=plt.cm.gray,extent=[times_motSQUID[0],times_motSQUID[-1], freqs_motSQUID[0], freqs_motSQUID[-1]],
           aspect='auto', origin='lower')

fig.colorbar(pos,ax=axs[0,1])
axs[0,1].set_ylim(global_ylim)
axs[0,1].set_xlim(global_mot_xlim)
axs[0,1].set_title('SQUID-MEG')                
axs[0,1].set_ylabel('Frequency (Hz)')


#OPM Y
opm_scale=0.2
power_fill_opmY[~np.isnan(power_fill_opmY)]=0
axs[1,0].imshow(outline_padY, cmap=plt.cm.gray,extent=[times[0],times[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower')

axs[1,0].set_ylim(global_ylim)
axs[1,0].set_xlim(global_xlim)
axs[1,0].set_title('4He-OPM$_{radial}$')                
axs[1,0].set_ylabel('Frequency (Hz)')

power_fill_motOPMY[~np.isnan(power_fill_motOPMY)]=0
axs[1,1].imshow(motopmY_outline, cmap=plt.cm.gray,extent=[times_motopm[0],times_motopm[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower')

fig.colorbar(pos,ax=axs[1,1])
axs[1,1].set_ylim(global_ylim)
axs[1,1].set_xlim(global_mot_xlim)
axs[1,1].set_title('4He-OPM$_{radial}$')                
axs[1,1].set_ylabel('Frequency (Hz)')


#OPM X
power_fill_opmX[~np.isnan(power_fill_opmX)]=0
axs[2,0].imshow(outline_padX, cmap=plt.cm.gray,extent=[times[0],times[-1], OPM_vis_TFR['x'][0].freqs[0], OPM_vis_TFR['x'][0].freqs[-1]],
           aspect='auto', origin='lower')

axs[2,0].set_ylim(global_ylim)
axs[2,0].set_xlim(global_xlim)
axs[2,0].set_title('4He-OPM$_{tangential}$')                
axs[2,0].set_xlabel('Time (s)')
axs[2,0].set_ylabel('Frequency (Hz)')

power_fill_motOPMX[~np.isnan(power_fill_motOPMX)]=0
axs[2,1].imshow(motopmX_outline, cmap=plt.cm.gray,extent=[times_motopm[0],times_motopm[-1], OPM_vis_TFR['y'][0].freqs[0], OPM_vis_TFR['y'][0].freqs[-1]],
           aspect='auto', origin='lower')

axs[2,1].set_ylim(global_ylim)
axs[2,1].set_xlim(global_mot_xlim)
fig.colorbar(pos,ax=axs[2,1])
axs[2,1].set_title('4He-OPM$_{radial}$')                
axs[2,1].set_xlabel('Time (s)')
axs[2,1].set_ylabel('Frequency (Hz)')

fig.set_figheight(10)
fig.set_figwidth(11)
fig.savefig(op.join(group_path,'Fig2A_Group_TFR_stats_Power_Blackoutline.pdf'))   
fig.savefig(op.join(group_path,'Fig2A_Group_TFR_stats_Power_Blackoutline.png'))   
 


#########################
# Plot burst rate/power #
#########################

print('SQUID: Loading data')
metrics_file = open(op.join(group_path, "SQUID_group_metrics.pkl"),'rb')
SQUID_metrics=pickle.load(metrics_file)

tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_MEG-ave.fif'))
vis_pwr_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_MEG-ave.fif'))
mot_pwr_times=tmp[0].times
btc_path = open(op.join(group_path, "SQUID_beta_timecourse.pkl"),'rb')
SQUID_beta_tc=pickle.load(btc_path)

#Calculate mean beta power over subjects
squid_vis_pwr=[]
squid_mot_pwr=[]
for s in SQUID_beta_tc['vis'].keys():
    squid_vis_pwr.append(SQUID_beta_tc['vis'][s])
for s in SQUID_beta_tc['mot'].keys():
    squid_mot_pwr.append(SQUID_beta_tc['mot'][s])    
    
squid_vis_pwr_av=np.mean(squid_vis_pwr,axis=0)    
squid_vis_pwr_sem=np.std(squid_vis_pwr,axis=0)/np.sqrt(np.shape(squid_vis_pwr)[0])
squid_mot_pwr_av=np.mean(squid_mot_pwr,axis=0)    
squid_mot_pwr_sem=np.std(squid_mot_pwr,axis=0)/np.sqrt(np.shape(squid_mot_pwr)[0])

squid_vis_pwr_md=np.median(squid_vis_pwr,axis=0)    
squid_mot_pwr_md=np.median(squid_mot_pwr,axis=0)    

#beta burst rate
squid_vis_br_av=SQUID_metrics['vis']['burst_rate_av']
squid_vis_br_sem=SQUID_metrics['vis']['burst_rate_sem']
squid_mot_br_av=SQUID_metrics['mot']['burst_rate_av']
squid_mot_br_sem=SQUID_metrics['mot']['burst_rate_sem']

#[STATS] Squid Burst rate

vis_br_times=SQUID_metrics['vis']['burst_rate_time']
mot_br_times=SQUID_metrics['mot']['burst_rate_time']
vis_time_sel_ix=np.where(((vis_br_times>=-.7) & (vis_br_times<=1.6))) #crop start to -.5 for vis
vis_br_times_crop=vis_br_times[vis_time_sel_ix]
mot_time_sel_ix=np.where(((mot_br_times>=-.75) & (mot_br_times<=1.2))) #crop to -.75, 1.2 for mot
mot_br_times_crop=mot_br_times[mot_time_sel_ix]

#cluster stat BR SQUID-VIS
X=np.squeeze(np.squeeze(SQUID_metrics['vis']['burst_rate'])[:,vis_time_sel_ix])

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, vis_br_clusters, vis_br_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat BR SQUID-MOT
X=np.squeeze(np.squeeze(SQUID_metrics['mot']['burst_rate'])[:,mot_time_sel_ix])

n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, mot_br_clusters, mot_br_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat PWR SQUID-VIS
#baseline power
vis_time_ix = np.where((vis_pwr_times >= vis_br_times_crop[0]) & (vis_pwr_times <= vis_br_times_crop[-1]))[0]
mot_time_ix = np.where((mot_pwr_times >= mot_br_times_crop[0]) & (mot_pwr_times <= mot_br_times_crop[-1]))[0]

baseline_ix = np.where((vis_pwr_times >= -0.5) & (vis_pwr_times <= -0.25))[0]
squid_vis_pwr_bl=np.zeros(np.shape(squid_vis_pwr))
squid_mot_pwr_bl=np.zeros(np.shape(squid_mot_pwr))
for s in range(len(squid_vis_pwr)):
    baseline_vis=np.mean(squid_vis_pwr[s][baseline_ix])
    squid_vis_pwr_bl[s,] = ((squid_vis_pwr[s] - baseline_vis) / baseline_vis)*100
    squid_mot_pwr_bl[s,] = ((squid_mot_pwr[s] - baseline_vis) / baseline_vis)*100

squid_vis_pwr_bl_crop=squid_vis_pwr_bl[:,vis_time_ix]
squid_mot_pwr_bl_crop=squid_mot_pwr_bl[:,mot_time_ix]
vis_pwr_times_crop=vis_pwr_times[vis_time_ix]
mot_pwr_times_crop=mot_pwr_times[mot_time_ix]

X=squid_vis_pwr_bl_crop

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, vis_pwr_clusters, vis_pwr_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat PWR SQUID-MOT
X=squid_mot_pwr_bl_crop

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, mot_pwr_clusters, mot_pwr_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

cw_br = "#4634eb"
cw_pwr = "#eb346b"

BETA_MEAN_VIS = np.mean(squid_vis_pwr_bl_crop,axis=0)
nsubs=np.shape(squid_vis_pwr_bl_crop)[0]
squid_vis_pwr_sem = np.std(squid_vis_pwr_bl_crop,axis=0)/np.sqrt(nsubs)
BETA_MEAN_MOT = np.mean(squid_mot_pwr_bl_crop,axis=0)
squid_mot_pwr_sem = np.std(squid_mot_pwr_bl_crop,axis=0)/np.sqrt(nsubs)

f, ax = plt.subplots(3, 2, figsize=(10, 4), gridspec_kw={"width_ratios":[3, 2.5]}, dpi=150, constrained_layout=True)
ax[0,0].plot(vis_pwr_times_crop, BETA_MEAN_VIS, lw=2, color=cw_pwr)
ax[0,0].fill_between(
    vis_pwr_times_crop,
    BETA_MEAN_VIS - squid_vis_pwr_sem,
    BETA_MEAN_VIS + squid_vis_pwr_sem,
    lw=0,
    alpha=0.2, 
    color=cw_pwr
)

squid_vis_br_av=np.average(np.squeeze(np.squeeze(SQUID_metrics['vis']['burst_rate'])[:,vis_time_sel_ix]),axis=0)
squid_vis_br_sem=np.std(np.squeeze(np.squeeze(SQUID_metrics['vis']['burst_rate'])[:,vis_time_sel_ix]),axis=0)/np.sqrt(nsubs)

ax[0,0].plot(vis_br_times_crop, 100*squid_vis_br_av, lw=2, color=cw_br)
ax[0,0].fill_between(
    vis_br_times_crop,
    100*(squid_vis_br_av - squid_vis_br_sem),
    100*(squid_vis_br_av + squid_vis_br_sem),
    lw=0,
    alpha=0.2, 
    color=cw_br,
)
ax[0,0].axvline(0, lw=1, c="black", linestyle="--")
ax[0,0].set_title('SQUID: Visual stimulation onset')
       

ax[0,1].plot(mot_pwr_times_crop, BETA_MEAN_MOT, lw=2, color=cw_pwr)
ax[0,1].fill_between(
    mot_pwr_times_crop,
    BETA_MEAN_MOT - squid_mot_pwr_sem,
    BETA_MEAN_MOT + squid_mot_pwr_sem,
    lw=0,
    alpha=0.2, 
    color=cw_pwr
)

squid_mot_br_av=np.average(np.squeeze(np.squeeze(SQUID_metrics['mot']['burst_rate'])[:,mot_time_sel_ix]),axis=0)
squid_mot_br_sem=np.std(np.squeeze(np.squeeze(SQUID_metrics['mot']['burst_rate'])[:,mot_time_sel_ix]),axis=0)/np.sqrt(nsubs)

ax[0,1].plot(mot_br_times_crop, 100*squid_mot_br_av, lw=2, color=cw_br)
ax[0,1].fill_between(
    mot_br_times_crop,
    100*(squid_mot_br_av - squid_mot_br_sem),
    100*(squid_mot_br_av + squid_mot_br_sem),
    lw=0,
    alpha=0.2, 
    color=cw_br
)


ax[0,1].axvline(0, lw=1, c="black", linestyle="--")
ax[0,1].set_title('SQUID: reach offset')

ax[0,0].set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
ax[0,1].set_xticks([-0.5, 0.0, 0.5, 1.0])
ax[0,1].set_yticks([])

for key, spine in ax[0,0].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)

for key, spine in ax[0,1].spines.items():
    if key in ["top", "left", "right"]:
        spine.set_visible(False)

ymin = np.min([ax[0,0].get_ylim()[0], ax[0,1].get_ylim()[0]])
ymax = np.max([ax[0,0].get_ylim()[1], ax[0,1].get_ylim()[1]])

y_pwr=ymin*1.1
y_br=ymin*1.2
#ADD STATS
#VIS-PWR
for i_c, cl in enumerate(vis_pwr_clusters):
    if vis_pwr_cluster_p_values[i_c] <= 0.05:
        sig=vis_pwr_times_crop[cl]        
        ax[0,0].hlines(y_pwr,sig[0],sig[-1],color=cw_pwr)

#VIS-PWR
for i_c, cl in enumerate(vis_br_clusters):
    if vis_br_cluster_p_values[i_c] <= 0.05:
        sig=vis_br_times_crop[cl]        
        ax[0,0].hlines(y_br,sig[0],sig[-1],color=cw_br)        

for i_c, cl in enumerate(mot_pwr_clusters):
    if mot_pwr_cluster_p_values[i_c] <= 0.05:
        sig=mot_pwr_times_crop[cl]        
        ax[0,1].hlines(y_pwr,sig[0],sig[-1],color=cw_pwr)

#VIS-PWR
for i_c, cl in enumerate(mot_br_clusters):
    if mot_br_cluster_p_values[i_c] <= 0.05:
        sig=mot_br_times_crop[cl]        
        ax[0,1].hlines(y_br,sig[0],sig[-1],color=cw_br)        
    
ax[0,0].set_xlim(-.7, 1.6)
ax[0,0].set_ylim(y_br*1.05,ymax)
ax[0,1].set_ylim(y_br*1.05,ymax)
ax[0,1].set_xlim(-.75, 1.2)


#OPM
print('OPM: Loading metrics')
opm_metrics_file = open(op.join(group_path, "OPM_group_metrics.pkl"),'rb')
OPM_metrics=pickle.load(opm_metrics_file)

#beta power timecourse
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_OPM-ave.fif'))
vis_pwr_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_OPM-ave.fif'))
mot_pwr_times=tmp[0].times
btc_path = open(op.join(group_path, "OPM_beta_timecourse.pkl"),'rb')
OPM_beta_tc=pickle.load(btc_path)

#Calculate mean beta power over subjects
opm_vis_pwr_x=[]
opm_vis_pwr_y=[]
opm_mot_pwr_x=[]
opm_mot_pwr_y=[]
for s in OPM_beta_tc['vis'].keys():    
    opm_vis_pwr_x.append(OPM_beta_tc['vis'][s]['x'])
    opm_vis_pwr_y.append(OPM_beta_tc['vis'][s]['y'])
for s in OPM_beta_tc['mot'].keys():
    opm_mot_pwr_x.append(OPM_beta_tc['mot'][s]['x'])
    opm_mot_pwr_y.append(OPM_beta_tc['mot'][s]['y'])

opm_vis_pwr_av_x=np.mean(opm_vis_pwr_x,axis=0)    
opm_vis_pwr_sem_x=np.std(opm_vis_pwr_x,axis=0)/np.sqrt(np.shape(opm_vis_pwr_x)[0])
opm_vis_pwr_av_y=np.mean(opm_vis_pwr_y,axis=0)    
opm_vis_pwr_sem_y=np.std(opm_vis_pwr_y,axis=0)/np.sqrt(np.shape(opm_vis_pwr_y)[0])
opm_mot_pwr_av_x=np.mean(opm_mot_pwr_x,axis=0)    
opm_mot_pwr_sem_x=np.std(opm_mot_pwr_x,axis=0)/np.sqrt(np.shape(opm_mot_pwr_x)[0])
opm_mot_pwr_av_y=np.mean(opm_mot_pwr_y,axis=0)    
opm_mot_pwr_sem_y=np.std(opm_mot_pwr_y,axis=0)/np.sqrt(np.shape(opm_mot_pwr_y)[0])

#Burst rate
opm_vis_br_av_x=OPM_metrics['x']['vis']['burst_rate_av']
opm_vis_br_sem_x=OPM_metrics['x']['vis']['burst_rate_sem']
opm_vis_br_av_y=OPM_metrics['y']['vis']['burst_rate_av']
opm_vis_br_sem_y=OPM_metrics['y']['vis']['burst_rate_sem']
opm_mot_br_av_x=OPM_metrics['x']['mot']['burst_rate_av']
opm_mot_br_sem_x=OPM_metrics['x']['mot']['burst_rate_sem']
opm_mot_br_av_y=OPM_metrics['y']['mot']['burst_rate_av']
opm_mot_br_sem_y=OPM_metrics['y']['mot']['burst_rate_sem']

#Stats
vis_br_times=OPM_metrics['x']['vis']['burst_rate_time']
mot_br_times=OPM_metrics['x']['mot']['burst_rate_time']
vis_time_sel_ix=np.where(((vis_br_times>=-.7) & (vis_br_times<=1.6))) #crop start to -.5 for vis
vis_br_times_crop=vis_br_times[vis_time_sel_ix]
mot_time_sel_ix=np.where(((mot_br_times>=-.75) & (mot_br_times<=1.2))) #crop to -.75, 1.2 for mot
mot_br_times_crop=mot_br_times[mot_time_sel_ix]

#cluster stat BR OPMX-VIS
X=np.squeeze(np.squeeze(OPM_metrics['x']['vis']['burst_rate'])[:,vis_time_sel_ix])

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

#cluster stat PWR OPMX-VIS
T_obs, vis_br_clusters, vis_br_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat BR OPX-MOT
X=np.squeeze(np.squeeze(OPM_metrics['x']['mot']['burst_rate'])[:,mot_time_sel_ix])

n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, mot_br_clusters, mot_br_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#prep for PWR stats
#baseline unaveraged data
vis_time_ix = np.where((vis_pwr_times >= vis_br_times_crop[0]) & (vis_pwr_times <= vis_br_times_crop[-1]))[0]
mot_time_ix = np.where((mot_pwr_times >= mot_br_times_crop[0]) & (mot_pwr_times <= mot_br_times_crop[-1]))[0]

baseline_ix = np.where((vis_pwr_times >= -0.5) & (vis_pwr_times <= -0.25))[0]
opmx_vis_pwr_bl=np.zeros(np.shape(opm_vis_pwr_x))
opmx_mot_pwr_bl=np.zeros(np.shape(opm_mot_pwr_x))
opmy_vis_pwr_bl=np.zeros(np.shape(opm_vis_pwr_y))
opmy_mot_pwr_bl=np.zeros(np.shape(opm_mot_pwr_y))
for s in range(len(opm_vis_pwr_x)):
    baseline_vis=np.mean(opm_vis_pwr_x[s][baseline_ix])
    opmx_vis_pwr_bl[s,] = ((opm_vis_pwr_x[s] - baseline_vis) / baseline_vis)*100
    opmx_mot_pwr_bl[s,] = ((opm_mot_pwr_x[s] - baseline_vis) / baseline_vis)*100
    baseline_vis=np.mean(opm_vis_pwr_y[s][baseline_ix])
    opmy_vis_pwr_bl[s,] = ((opm_vis_pwr_y[s] - baseline_vis) / baseline_vis)*100
    opmy_mot_pwr_bl[s,] = ((opm_mot_pwr_y[s] - baseline_vis) / baseline_vis)*100

opmx_vis_pwr_bl_crop=opmx_vis_pwr_bl[:,vis_time_ix]
opmx_mot_pwr_bl_crop=opmx_mot_pwr_bl[:,mot_time_ix]
opmy_vis_pwr_bl_crop=opmy_vis_pwr_bl[:,vis_time_ix]
opmy_mot_pwr_bl_crop=opmy_mot_pwr_bl[:,mot_time_ix]
vis_pwr_times_crop=vis_pwr_times[vis_time_ix]
mot_pwr_times_crop=mot_pwr_times[mot_time_ix]

X=opmx_vis_pwr_bl_crop

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, vis_pwr_clusters, vis_pwr_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat PWR SQUID-MOT
X=opmx_mot_pwr_bl_crop

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, mot_pwr_clusters, mot_pwr_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)


#plot
cw_br = "#4634eb"
cw_pwr = "#eb346b"

BETA_MEAN_VIS_X = np.mean(opmx_vis_pwr_bl_crop,axis=0)
BETA_MEAN_VIS_Y = np.mean(opmy_vis_pwr_bl_crop,axis=0)
nsubs=np.shape(opmx_vis_pwr_bl_crop)[0]
opmx_vis_pwr_sem = np.std(opmx_vis_pwr_bl_crop,axis=0)/np.sqrt(nsubs)
opmy_vis_pwr_sem = np.std(opmy_vis_pwr_bl_crop,axis=0)/np.sqrt(nsubs)
BETA_MEAN_MOT_X = np.mean(opmx_mot_pwr_bl_crop,axis=0)
BETA_MEAN_MOT_Y = np.mean(opmy_mot_pwr_bl_crop,axis=0)
opmx_mot_pwr_sem = np.std(opmx_mot_pwr_bl_crop,axis=0)/np.sqrt(nsubs)
opmy_mot_pwr_sem = np.std(opmy_mot_pwr_bl_crop,axis=0)/np.sqrt(nsubs)


#X
ax[2,0].plot(vis_pwr_times_crop, BETA_MEAN_VIS_X, lw=2, color=cw_pwr)
ax[2,0].fill_between(
    vis_pwr_times[vis_time_ix],    
    BETA_MEAN_VIS_X - opmx_vis_pwr_sem,
    BETA_MEAN_VIS_X + opmx_vis_pwr_sem,
    lw=0,
    alpha=0.2, 
    color=cw_pwr
)

opmx_vis_br_av = np.squeeze(np.average(np.squeeze(OPM_metrics['x']['vis']['burst_rate'])[:,vis_time_sel_ix],axis=0))
opmx_vis_br_sem = np.squeeze(np.std(np.squeeze(OPM_metrics['x']['vis']['burst_rate'])[:,vis_time_sel_ix],axis=0))/np.sqrt(nsubs)

ax[2,0].plot(vis_br_times_crop, 100*opmx_vis_br_av, lw=2, color=cw_br)
ax[2,0].fill_between(
    vis_br_times_crop,
    100*(opmx_vis_br_av - opmx_vis_br_sem),
    100*(opmx_vis_br_av + opmx_vis_br_sem),
    lw=0,
    alpha=0.2, 
    color=cw_br
)
ax[2,0].axvline(0, lw=1, c="black", linestyle="--")
ax[2,0].set_title('OPM$_{tangential}$: Visual stimulation onset')


ax[2,1].plot(mot_pwr_times_crop, BETA_MEAN_MOT_X, lw=2, color=cw_pwr)
ax[2,1].fill_between(
    mot_pwr_times_crop,
    BETA_MEAN_MOT_X - opmx_mot_pwr_sem,
    BETA_MEAN_MOT_X + opmx_mot_pwr_sem,
    lw=0,
    alpha=0.2, 
    color=cw_pwr
)

opmx_mot_br_av = np.squeeze(np.average(np.squeeze(OPM_metrics['x']['mot']['burst_rate'])[:,mot_time_sel_ix],axis=0))
opmx_mot_br_sem = np.squeeze(np.std(np.squeeze(OPM_metrics['x']['mot']['burst_rate'])[:,mot_time_sel_ix],axis=0))/np.sqrt(nsubs)

ax[2,1].plot(mot_br_times_crop, 100*opmx_mot_br_av, lw=2, color=cw_br)
ax[2,1].fill_between(
    mot_br_times_crop,
    100*(opmx_mot_br_av - opmx_mot_br_sem),
    100*(opmx_mot_br_av + opmx_mot_br_sem),
    lw=0,
    alpha=0.2, 
    color=cw_br
)
ax[2,1].axvline(0, lw=1, c="black", linestyle="--")
ax[2,1].set_title('OPM$_{tangential}$: reach offset')

ax[2,0].set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
ax[2,1].set_xticks([-0.5, 0.0, 0.5, 1.0])
ax[2,1].set_yticks([])

for key, spine in ax[2,0].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)

for key, spine in ax[2,1].spines.items():
    if key in ["top", "left", "right"]:
        spine.set_visible(False)

ymin = np.min([ax[2,0].get_ylim()[0], ax[2,1].get_ylim()[0]])
ymax = np.max([ax[2,0].get_ylim()[1], ax[2,1].get_ylim()[1]])
    
y_pwr=ymin*1.1
y_br=ymin*1.2
#ADD STATS
#VIS-PWR
for i_c, cl in enumerate(vis_pwr_clusters):
    if vis_pwr_cluster_p_values[i_c] <= 0.05:
        sig=vis_pwr_times_crop[cl]        
        ax[2,0].hlines(y_pwr,sig[0],sig[-1],color=cw_pwr)

#VIS-PWR
for i_c, cl in enumerate(vis_br_clusters):
    if vis_br_cluster_p_values[i_c] <= 0.05:
        sig=vis_br_times_crop[cl]        
        ax[2,0].hlines(y_br,sig[0],sig[-1],color=cw_br)        

for i_c, cl in enumerate(mot_pwr_clusters):
    if mot_pwr_cluster_p_values[i_c] <= 0.05:
        sig=mot_pwr_times_crop[cl]        
        ax[2,1].hlines(y_pwr,sig[0],sig[-1],color=cw_pwr)

#VIS-PWR
for i_c, cl in enumerate(mot_br_clusters):
    if mot_br_cluster_p_values[i_c] <= 0.05:
        sig=mot_br_times_crop[cl]        
        ax[2,1].hlines(y_br,sig[0],sig[-1],color=cw_br)        
    
ax[2,0].set_xlim(-.7, 1.6)
ax[2,0].set_ylim(y_br*1.05,ymax)
ax[2,1].set_ylim(y_br*1.05,ymax)
ax[2,1].set_xlim(-.75, 1.2)

#Y
#cluster stat BR OPMY-VIS
X=np.squeeze(np.squeeze(OPM_metrics['y']['vis']['burst_rate'])[:,vis_time_sel_ix])

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

#cluster stat PWR OPMX-VIS
T_obs, vis_br_clusters, vis_br_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat BR OPMY-MOT
X=np.squeeze(np.squeeze(OPM_metrics['y']['mot']['burst_rate'])[:,mot_time_sel_ix])

n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, mot_br_clusters, mot_br_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat PWR OPMY-VIS
X=opmy_vis_pwr_bl_crop

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, vis_pwr_clusters, vis_pwr_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

#cluster stat PWR OPMY-MOT
X=opmy_mot_pwr_bl_crop

alpha_cluster_forming = 0.05
n_observations = len(X)
pval = alpha_cluster_forming
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

T_obs, mot_pwr_clusters, mot_pwr_cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
X,
n_permutations=5000,
threshold=thresh,
tail=0,
n_jobs=None,
out_type="mask",
)

ax[1,0].plot(vis_pwr_times[vis_time_ix], BETA_MEAN_VIS_Y, lw=2, color=cw_pwr)
ax[1,0].fill_between(
    vis_pwr_times_crop,
    BETA_MEAN_VIS_Y - opmy_vis_pwr_sem,
    BETA_MEAN_VIS_Y + opmy_vis_pwr_sem,
    lw=0,
    alpha=0.2, 
    color=cw_pwr
)

opmy_vis_br_av = np.squeeze(np.average(np.squeeze(OPM_metrics['y']['vis']['burst_rate'])[:,vis_time_sel_ix],axis=0))
opmy_vis_br_sem = np.squeeze(np.std(np.squeeze(OPM_metrics['y']['vis']['burst_rate'])[:,vis_time_sel_ix],axis=0))/np.sqrt(nsubs)
ax[1,0].plot(vis_br_times_crop, 100*opmy_vis_br_av, lw=2, color=cw_br)
ax[1,0].fill_between(
    vis_br_times_crop,
    100*(opmy_vis_br_av - opmy_vis_br_sem),
    100*(opmy_vis_br_av + opmy_vis_br_sem),
    lw=0,
    alpha=0.2, 
    color=cw_br
)
ax[1,0].axvline(0, lw=1, c="black", linestyle="--")
ax[1,0].set_title('OPM$_{radial}$: Visual stimulation onset')


ax[1,1].plot(mot_pwr_times_crop, BETA_MEAN_MOT_Y, lw=2, color=cw_pwr)
ax[1,1].fill_between(
    mot_pwr_times_crop,
    BETA_MEAN_MOT_Y - opmy_mot_pwr_sem,
    BETA_MEAN_MOT_Y + opmy_mot_pwr_sem,
    lw=0,
    alpha=0.2, 
    color=cw_pwr
)

opmy_mot_br_av = np.squeeze(np.average(np.squeeze(OPM_metrics['y']['mot']['burst_rate'])[:,mot_time_sel_ix],axis=0))
opmy_mot_br_sem = np.squeeze(np.std(np.squeeze(OPM_metrics['y']['mot']['burst_rate'])[:,mot_time_sel_ix],axis=0))/np.sqrt(nsubs)

ax[1,1].plot(mot_br_times_crop, 100*opmy_mot_br_av, lw=2, color=cw_br)
ax[1,1].fill_between(
    mot_br_times_crop,
    100*(opmy_mot_br_av - opmy_mot_br_sem),
    100*(opmy_mot_br_av + opmy_mot_br_sem),
    lw=0,
    alpha=0.2, 
    color=cw_br
)
ax[1,1].axvline(0, lw=1, c="black", linestyle="--")
ax[1,1].set_title('OPM$_{radial}$: reach offset')

ax[1,0].set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
ax[1,1].set_xticks([-0.5, 0.0, 0.5, 1.0])
ax[1,1].set_yticks([])

for key, spine in ax[1,0].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)

for key, spine in ax[1,1].spines.items():
    if key in ["top", "left", "right"]:
        spine.set_visible(False)

ymin = np.min([ax[1,0].get_ylim()[0], ax[1,1].get_ylim()[0]])
ymax = np.max([ax[1,0].get_ylim()[1], ax[1,1].get_ylim()[1]])
    
#ADD STATS
#VIS-PWR
for i_c, cl in enumerate(vis_pwr_clusters):
    if vis_pwr_cluster_p_values[i_c] <= 0.05:
        sig=vis_pwr_times_crop[cl]        
        ax[1,0].hlines(y_pwr,sig[0],sig[-1],color=cw_pwr)

#VIS-PWR
for i_c, cl in enumerate(vis_br_clusters):
    if vis_br_cluster_p_values[i_c] <= 0.05:
        sig=vis_br_times_crop[cl]        
        ax[1,0].hlines(y_br,sig[0],sig[-1],color=cw_br)        

for i_c, cl in enumerate(mot_pwr_clusters):
    if mot_pwr_cluster_p_values[i_c] <= 0.05:
        sig=mot_pwr_times_crop[cl]        
        ax[1,1].hlines(y_pwr,sig[0],sig[-1],color=cw_pwr)

#VIS-PWR
for i_c, cl in enumerate(mot_br_clusters):
    if mot_br_cluster_p_values[i_c] <= 0.05:
        sig=mot_br_times_crop[cl]        
        ax[1,1].hlines(y_br,sig[0],sig[-1],color=cw_br)        
    
ax[1,0].set_xlim(-.7, 1.6)
ax[1,0].set_ylim(y_br*1.05,ymax)
ax[1,1].set_ylim(y_br*1.05,ymax)
ax[1,1].set_xlim(-.75, 1.2)

f.set_figheight(10)
f.set_figwidth(11)
f.savefig(op.join(group_path,'Fig2B_BurstratePower.pdf'))   
f.savefig(op.join(group_path,'Fig2B_BurstratePower.png'))  