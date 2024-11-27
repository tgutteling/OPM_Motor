import pickle
import os.path as op
import numpy as np
from os import listdir
import mne

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'

beta_range=[15, 30]

sens_av=0 #average over sensors or not

#SQUID
VIS_SQUID_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'MEG_VIS_sl_ga' in d]
MOT_SQUID_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'MEG_MOT_sl_ga' in d]
VIS_SQUID_TFRs.sort()
MOT_SQUID_TFRs.sort()

print('SQUID: Loading Visual TFR data')
squid_beta_tc={'vis':[], 'mot':[]}
vis=dict()
for s in VIS_SQUID_TFRs:
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    beta_inds=np.where((tmp.freqs >= beta_range[0]) & (tmp.freqs <= beta_range[1]))[0]
    if sens_av:
        vis[pp_ix]=np.mean(np.mean(tmp.data[:,beta_inds,:],axis=1),axis=0)
    else:
        vis[pp_ix]=np.mean(tmp.data[:,beta_inds,:],axis=1)
squid_beta_tc['vis']=vis
    
print('SQUID: Loading Motor TFR data')
mot=dict()
for s in MOT_SQUID_TFRs:
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    beta_inds=np.where((tmp.freqs >= beta_range[0]) & (tmp.freqs <= beta_range[1]))[0]
    if sens_av:
        mot[pp_ix]=np.mean(np.mean(tmp.data[:,beta_inds,:],axis=1),axis=0)
    else:
        mot[pp_ix]=np.mean(tmp.data[:,beta_inds,:],axis=1)
squid_beta_tc['mot']=mot

print('Saving SQUID timecourses')       
if sens_av:
    file = op.join(group_path, "SQUID_beta_timecourse.pkl")
else:
    file = op.join(group_path, "SQUID_beta_timecourse_perSens.pkl")    
pickle.dump(squid_beta_tc, open(file, "wb"))
print('Done.')              

#OPM
VIS_OPM_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'OPM_VIS_sl_noev' in d]
MOT_OPM_TFRs = [d for d in listdir(op.join(group_path,'TFR')) if 'OPM_MOT_sl_noev' in d]
VIS_OPM_TFRs.sort()
MOT_OPM_TFRs.sort()

print('OPM: Loading Visual TFR data')
opm_beta_tc={'vis':[], 'mot':[]}
vis=dict()
#tc=dict()
for s in VIS_OPM_TFRs:
    tc=dict({'x':[], 'y':[], 'z':[]})
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    beta_inds=np.where((tmp.freqs >= beta_range[0]) & (tmp.freqs <= beta_range[1]))[0]
    picks_x=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*x')
    picks_y=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*y')
    picks_z=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*z')
    if sens_av:
        tc['x']=np.mean(np.mean(tmp.copy().pick(picks=picks_x).data[:,beta_inds,:],axis=1),axis=0)            
        tc['y']=np.mean(np.mean(tmp.copy().pick(picks=picks_y).data[:,beta_inds,:],axis=1),axis=0)
        tc['z']=np.mean(np.mean(tmp.copy().pick(picks=picks_z).data[:,beta_inds,:],axis=1),axis=0)
    else:
        tc['x']=np.mean(tmp.copy().pick(picks=picks_x).data[:,beta_inds,:],axis=1) 
        tc['y']=np.mean(tmp.copy().pick(picks=picks_y).data[:,beta_inds,:],axis=1)
        tc['z']=np.mean(tmp.copy().pick(picks=picks_z).data[:,beta_inds,:],axis=1)
    vis[pp_ix]=tc
opm_beta_tc['vis']=vis

print('OPM: Loading Motor TFR data')
mot=dict()
#tc=dict()
for s in MOT_OPM_TFRs:
    tc=dict({'x':[], 'y':[], 'z':[]})
    print('Loading ' +s)
    pp_ix=s[:2]
    tmp=mne.time_frequency.read_tfrs(op.join(group_path,'TFR',s))
    beta_inds=np.where((tmp.freqs >= beta_range[0]) & (tmp.freqs <= beta_range[1]))[0]
    picks_x=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*x')
    picks_y=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*y')
    picks_z=mne.pick_channels_regexp(tmp.info['ch_names'],'^.*z')
    if sens_av:
        tc['x']=np.mean(np.mean(tmp.copy().pick(picks=picks_x).data[:,beta_inds,:],axis=1),axis=0)
        tc['y']=np.mean(np.mean(tmp.copy().pick(picks=picks_y).data[:,beta_inds,:],axis=1),axis=0)
        tc['z']=np.mean(np.mean(tmp.copy().pick(picks=picks_z).data[:,beta_inds,:],axis=1),axis=0)
    else:        
        tc['x']=np.mean(tmp.copy().pick(picks=picks_x).data[:,beta_inds,:],axis=1)
        tc['y']=np.mean(tmp.copy().pick(picks=picks_y).data[:,beta_inds,:],axis=1)
        tc['z']=np.mean(tmp.copy().pick(picks=picks_z).data[:,beta_inds,:],axis=1)
    mot[pp_ix]=tc
opm_beta_tc['mot']=mot

print('Saving OPM timecourses')
if sens_av:
    file = op.join(group_path, "OPM_beta_timecourse.pkl")
else:       
    file = op.join(group_path, "OPM_beta_timecourse_perSens.pkl")
pickle.dump(opm_beta_tc, open(file, "wb"))
print('All done.')
