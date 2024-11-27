#Figure 6 - SNR & Component specific Percent baseline change

import pickle
import numpy as np
import os.path as op
import mne
import matplotlib.pyplot as plt
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'

#OPM
btc_path = open(op.join(group_path, "OPM_beta_timecourse_perSens.pkl"),'rb')
OPM_beta_tc=pickle.load(btc_path)

tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_OPM-ave.fif'))
vis_pwr_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_OPM-ave.fif'))
mot_pwr_times=tmp[0].times


#SNR per subject
baseline_ix={c: [] for c in ['vis','mot']}
signal_ix={c: [] for c in ['vis','mot']}

bl=[-.5,0]
sig=[.1,1]
baseline_ix['vis'] = np.where((vis_pwr_times >= bl[0]) & (vis_pwr_times <= bl[1]))[0]
signal_ix['vis'] = np.where((vis_pwr_times >= sig[0]) & (vis_pwr_times <= sig[1]))[0]
baseline_ix['mot'] = np.where((mot_pwr_times >= bl[0]) & (mot_pwr_times <= bl[1]))[0]
signal_ix['mot'] = np.where((mot_pwr_times >= sig[0]) & (mot_pwr_times <= sig[1]))[0]

subs=[s for s in OPM_beta_tc['vis'].keys()]
OPM_SNR={c: {ax: {s:[] for s in subs} for ax in ['x','y','z']}  for c in ['vis','mot']}
for c in OPM_beta_tc.keys():
    for s in OPM_beta_tc[c].keys():
        for ax in OPM_beta_tc[c][s].keys():
            tmp=OPM_beta_tc[c][s][ax]
        
            tmp=tmp.T-np.average(tmp[:,baseline_ix[c]],axis=1) #apply baseline
            
            signal=np.max(np.abs(tmp[signal_ix[c],]),axis=0)
            noise=np.std(tmp[baseline_ix[c],],axis=0)            
            OPM_SNR[c][ax][s]=np.mean(signal/noise)



#SQUID
btc_path = open(op.join(group_path, "SQUID_beta_timecourse_perSens.pkl"),'rb')
SQUID_beta_tc=pickle.load(btc_path)

tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_MEG-ave.fif'))
vis_pwr_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_MEG-ave.fif'))
mot_pwr_times=tmp[0].times

#SNR per subject
baseline_ix={c: [] for c in ['vis','mot']}
signal_ix={c: [] for c in ['vis','mot']}

baseline_ix['vis'] = np.where((vis_pwr_times >= -0.5) & (vis_pwr_times <= -0.25))[0]
signal_ix['vis'] = np.where((vis_pwr_times >= 0.1) & (vis_pwr_times <= 1.5))[0]
baseline_ix['mot'] = np.where((mot_pwr_times >= -0.5) & (mot_pwr_times <= -0.25))[0]
signal_ix['mot'] = np.where((mot_pwr_times >= 0.1) & (mot_pwr_times <= 1))[0]

subs=[s for s in SQUID_beta_tc['vis'].keys()]
SQUID_SNR={c: {s : [] for s in subs} for c in ['vis','mot']}
for c in SQUID_beta_tc.keys():
    for s in SQUID_beta_tc[c].keys():
        tmp=SQUID_beta_tc[c][s]
        
        tmp=tmp.T-np.average(tmp[:,baseline_ix[c]],axis=1) #apply baseline
        
        signal=np.max(np.abs(tmp[signal_ix[c],]),axis=0)
        noise=np.std(tmp[baseline_ix[c],],axis=0)        
        SQUID_SNR[c][s]=np.mean(signal/noise)             
        
#plot       
titles=['SQUID','OPM$_{radial}$','OPM$_{tangential}$']
xlabels=['SQUID','OPM$_{radial}$','OPM$_{tangential}$']
ylabels=['SNR (signal max / std baseline)']   
 
data=np.vstack((list(SQUID_SNR['vis'].values()),list(OPM_SNR['vis']['y'].values()),list(OPM_SNR['vis']['x'].values()))).transpose()

fig,ax=plt.subplots(1,1)
fig.set_figheight(5)
fig.set_figwidth(8)
vlp=ax.violinplot(data,showmeans=True)
ax.set_ylabel(ylabels[0])
ax.set_xlabel('Sensors')
ax.set_xticks(1+np.arange(len(xlabels)),labels=xlabels)
vlp['cmeans'].set_color('k') 
vlp['cmaxes'].set_color('k')
vlp['cmins'].set_color('k')
vlp['cbars'].set_color('k')
vlp['bodies'][0].set_facecolor('k')
vlp['bodies'][1].set_facecolor('k')
vlp['bodies'][2].set_facecolor('k')     
fig.savefig(op.join(group_path,'Fig6A.png'))          
fig.savefig(op.join(group_path,'Fig6A.pdf'))          

################
# PrC BASELINE #
################

print('Loading SQUID burst scores')        
vis_tw_file = open(op.join(group_path, "SQUID_Visual_timewindow.pkl"),'rb')
SQUID_tw=pickle.load(vis_tw_file)

print('Loading OPM burst scores')        
vis_tw_file = open(op.join(group_path, "OPMx_Visual_timewindow.pkl"),'rb')
OPMx_tw=pickle.load(vis_tw_file)
vis_tw_file = open(op.join(group_path, "OPMy_Visual_timewindow.pkl"),'rb')
OPMy_tw=pickle.load(vis_tw_file)

print('Loading group metrics')
metrics_file = open(op.join(group_path, "OPM_group_metrics.pkl"),'rb')
opm_group_metrics=pickle.load(metrics_file)
metrics_file = open(op.join(group_path, "SQUID_group_metrics.pkl"),'rb')
squid_group_metrics=pickle.load(metrics_file)

#common keys are PC_6, PC_7, PC_8

PCs_to_plot=['PC_6', 'PC_7', 'PC_8']
br_av_x=opm_group_metrics['x']['vis']['burst_rate_av']
br_av_y=opm_group_metrics['y']['vis']['burst_rate_av']
br_av_sq=squid_group_metrics['vis']['burst_rate_av']

opm_time=opm_group_metrics['x']['vis']['burst_rate_time']
squid_time=opm_group_metrics['x']['vis']['burst_rate_time']

#get averages for burst rate (all waveforms)
tw=[.5,1.5]
tw_ix= np.where((opm_time >= tw[0]) & (opm_time <= tw[1]))[0]
br_av_x=np.nanmean(br_av_x[tw_ix])*100
br_av_y=np.nanmean(br_av_y[tw_ix])*100
tw_ix= np.where((squid_time >= tw[0]) & (squid_time <= tw[1]))[0]
br_av_sq=np.nanmean(br_av_sq[tw_ix])*100

data={PC:[] for PC in PCs_to_plot}
for PC in PCs_to_plot:
    dat_SQ=SQUID_tw[PC]
    dat_OPMX=OPMx_tw[PC]
    dat_OPMY=OPMy_tw[PC]

    data[PC]=np.vstack((dat_SQ[3,]-br_av_sq,dat_OPMY[3,]-br_av_y,dat_OPMX[3,]-br_av_x)).transpose()
       
    
data_all=np.hstack((data['PC_6'],data['PC_7'],data['PC_8']))    
pos=[1,2,3,5,6,7,9,10,11]
fig,ax=plt.subplots(1,1)
fig.set_figheight(5)
fig.set_figwidth(8)
vlp=ax.violinplot(data_all,pos,showmeans=True)
ax.set_title('Percent baseline change PC 6 / PC 7 / PC 8')
ax.set_ylabel('%Change from baseline')
ax.set_xlabel('Sensors')
ax.set_xticks(pos,labels=['Squid','OPM$_{radial}$','OPM$_{tangential}$','Squid','OPM$_{radial}$','OPM$_{tangential}$','Squid','OPM$_{radial}$','OPM$_{tangential}$'])
vlp['cmeans'].set_color('k')
vlp['cmaxes'].set_color('k')
vlp['cmins'].set_color('k')
vlp['cbars'].set_color('k')
for i in range(3): 
    vlp['bodies'][i].set_facecolor('b')
    vlp['bodies'][i+3].set_facecolor('g')
    vlp['bodies'][i+6].set_facecolor('r')    

plt.xticks(rotation=-45)
    
plt.savefig(op.join(group_path,'Fig6B.png'))
plt.savefig(op.join(group_path,'Fig6B.pdf'))
