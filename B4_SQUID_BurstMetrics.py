import pickle
import os.path as op
import numpy as np
from os import listdir
import mne
from scipy.ndimage import gaussian_filter

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'

print('Loading data')
#Get subject files and sort by type
VIS_SQUID_bursts = [d for d in listdir(op.join(group_path,'bursts')) if 'SQUID_VIS' in d]
MOT_SQUID_bursts = [d for d in listdir(op.join(group_path,'bursts')) if 'SQUID_MOT' in d]
VIS_SQUID_bursts.sort()
MOT_SQUID_bursts.sort()

print('Getting SQUID waveforms')
waveforms = []
for vis_file in VIS_SQUID_bursts:
    file = open(op.join(group_path,'bursts',vis_file), 'rb')
    bs = pickle.load(file)    
    for i in range(len(bs)):
        am_bs = len(bs[i]["waveform"])
        bs_samp_ix = np.random.choice(np.arange(am_bs), int(am_bs))
        bs_samp = np.vstack(bs[i]['waveform'])[bs_samp_ix, :]
        waveforms.append(bs_samp)        
    
waveforms = np.vstack(waveforms)
waveforms_medians = np.median(waveforms, axis=1)

print('Collecting subject metrics')
#create subject metrics
sub_metrics = {}
for (vis,mot) in zip(VIS_SQUID_bursts, MOT_SQUID_bursts):
    pp_ix1=vis[:2]    
    pp_ix2=mot[:2]
    if not pp_ix1==pp_ix2:
        print('File order mismatch, please check burst files')
        break
    file = open(op.join(group_path,'bursts',vis), 'rb')
    bs_vis = pickle.load(file)    
    file = open(op.join(group_path,'bursts',mot), 'rb')
    bs_mot = pickle.load(file)        
    
    metrics = {
        "vis": {
            "waveform": [],
            "peak_time": [],
            "peak_freq": [],
            "peak_amp_base": [],
            "fwhm_freq": [],
            "fwhm_time": [],
            "peak_adjustment": [],
            "trial": [],
            "pp_ix": [],           
        },
        "mot": {
            "waveform": [],
            "peak_time": [],
            "peak_freq": [],
            "peak_amp_base": [],
            "fwhm_freq": [],
            "fwhm_time": [],
            "peak_adjustment": [],
            "trial": [],
            "pp_ix": [],
        }
    }
    #VIS:
    #collapse over channels
    wf_vis=[]
    for ch in range(len(bs_vis)):
        wf_vis.append(np.array(bs_vis[ch]['waveform']))
    wf_vis=np.vstack(wf_vis)
    
    #clear top and bottom 1 percentile (overall)
    wf_median = np.median(wf_vis, axis=1)
    wf_ixs = np.where(
        (wf_median > np.percentile(waveforms_medians, 1)) & 
        (wf_median < np.percentile(waveforms_medians, 99)))[0]
    wf_vis = wf_vis[wf_ixs,:]
    
    #fill in the metrics, omitting omitted bursts    
    metrics['vis']["waveform"].append(wf_vis)
    metrics['vis']["waveform"]=np.vstack(metrics['vis']["waveform"])
    for k in ["peak_time", "peak_amp_base", "fwhm_freq", "fwhm_time", "peak_freq", "trial"]:
        tmp=[]
        for ch in range(len(bs_vis)):
            tmp.append(np.array(bs_vis[ch][k]))
        tmp=np.hstack(tmp)
        metrics['vis'][k].append(tmp[wf_ixs])      
        metrics['vis'][k]=np.vstack(metrics['vis'][k])
    metrics['vis']['pp_ix']=pp_ix1
    
    #MOT:
    #collapse over channels
    wf_mot=[]
    for ch in range(len(bs_mot)):
        wf_mot.append(np.array(bs_mot[ch]['waveform']))
    wf_mot=np.vstack(wf_mot)
    
    #clear top and bottom 1 percentile (overall)
    wf_median = np.median(wf_mot, axis=1)
    wf_ixs = np.where(
        (wf_median > np.percentile(waveforms_medians, 1)) & 
        (wf_median < np.percentile(waveforms_medians, 99)))[0]
    wf_mot = wf_mot[wf_ixs,:]
    
    #fill in the metrics, omitting omitted bursts    
    metrics['mot']["waveform"].append(wf_mot)
    metrics['mot']["waveform"]=np.vstack(metrics['mot']["waveform"])
    for k in ["peak_time", "peak_amp_base", "fwhm_freq", "fwhm_time", "peak_freq", "trial"]:
        tmp=[]
        for ch in range(len(bs_mot)):
            tmp.append(np.array(bs_mot[ch][k]))
        tmp=np.hstack(tmp)
        metrics['mot'][k].append(tmp[wf_ixs])      
        metrics['mot'][k]=np.vstack(metrics['mot'][k])
    metrics['mot']['pp_ix']=pp_ix2
    
    sub_metrics[pp_ix1] = metrics

#save metrics
print('Saving subject metrics')
sub_metrics_path = op.join(group_path, "SQUID_sub_metrics.pkl")
pickle.dump(sub_metrics, open(sub_metrics_path, "wb"))

#Now create group averages
#We need to know the trial timecourse, so load an evoked for both conditions and extract times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_MEG-ave.fif'))
vis_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_MEG-ave.fif'))
mot_times=tmp[0].times

print('aggregating metrics')
vis_wf=np.zeros([len(sub_metrics.keys()),np.shape(sub_metrics['01']['vis']['waveform'])[1]])
mot_wf=np.zeros([len(sub_metrics.keys()),np.shape(sub_metrics['01']['mot']['waveform'])[1]])
vis_burst_rate = []
mot_burst_rate = []
vis_peak_freq = []
mot_peak_freq = []
vis_peak_amp = []
mot_peak_amp = []
vis_fwhm_time = []
mot_fwhm_time = []
vis_fwhm_freq = []
mot_fwhm_freq = []
for ind,s in enumerate(sub_metrics.keys()):
    vis_wf[ind,]=np.average(sub_metrics[s]['vis']['waveform'],axis=0)
    mot_wf[ind,]=np.average(sub_metrics[s]['mot']['waveform'],axis=0)

    #burst rate
    buffer = 0.125
    bin_width=0.05
    baseline_range = [-0.5, -0.25]
    visual_time_bins = np.arange(vis_times[0] + buffer, vis_times[-1] - buffer, bin_width)
    vis_hist, t_bin_edges = np.histogram(sub_metrics[s]['vis']['peak_time'],bins=visual_time_bins)
    vis_hist = vis_hist / bin_width
    br_vis=gaussian_filter(vis_hist, 1)
    
    motor_time_bins = np.arange(mot_times[0] + buffer, mot_times[-1] - buffer, bin_width)
    mot_hist, t_bin_edges = np.histogram(sub_metrics[s]['mot']['peak_time'],bins=motor_time_bins)
    mot_hist = mot_hist / bin_width
    br_mot=gaussian_filter(mot_hist, 1)
    
    #now baseline burst rate    
    vis_time_plot = visual_time_bins[:-1]
    mot_time_plot = motor_time_bins[:-1]
    baseline_ixs = np.where(
        (vis_time_plot >= baseline_range[0]) &
        (vis_time_plot <= baseline_range[-1]))
    base_bursts = np.mean(br_vis[baseline_ixs])
    br_vis = (br_vis - base_bursts) / base_bursts
    br_mot = (br_mot - base_bursts) / base_bursts        
    
    vis_burst_rate.append(br_vis)
    mot_burst_rate.append(br_mot)
        
    #peak_freq: peak frequency distribution
    vis_peak_freq.append(sub_metrics[s]['vis']['peak_freq'])
    mot_peak_freq.append(sub_metrics[s]['mot']['peak_freq'])
    
    #peak_amp_base: peak amplitude distribution
    vis_peak_amp.append(sub_metrics[s]['vis']['peak_amp_base'])
    mot_peak_amp.append(sub_metrics[s]['mot']['peak_amp_base'])
    
    #fwhm_time: burst duration distribution
    vis_fwhm_time.append(sub_metrics[s]['vis']['fwhm_time'])
    mot_fwhm_time.append(sub_metrics[s]['mot']['fwhm_time'])
        
    #fwhm_freq: frequency span distribution
    vis_fwhm_freq.append(sub_metrics[s]['vis']['fwhm_freq'])
    mot_fwhm_freq.append(sub_metrics[s]['mot']['fwhm_freq'])
    
print('Everything sorted.')    

#calculate burst rate average and SEM
print('Elaborating burst rate')
brate_mean_vis = np.mean(vis_burst_rate, axis=0)
brate_sem_vis = np.std(vis_burst_rate, axis=0) / np.sqrt(np.shape(vis_burst_rate)[0])
brate_mean_mot = np.mean(mot_burst_rate, axis=0)
brate_sem_mot = np.std(mot_burst_rate, axis=0) / np.sqrt(np.shape(vis_burst_rate)[0])


#collapse all data for the histograms
print('Flattening some stuff')
vis_peak_freq=np.hstack(vis_peak_freq).flatten()
mot_peak_freq=np.hstack(mot_peak_freq).flatten()
vis_peak_amp=np.hstack(vis_peak_amp).flatten()
mot_peak_amp=np.hstack(mot_peak_amp).flatten()
mot_peak_freq=np.hstack(mot_peak_freq).flatten()
vis_fwhm_time=np.hstack(vis_fwhm_time).flatten()
mot_fwhm_time=np.hstack(mot_fwhm_time).flatten()
vis_fwhm_freq=np.hstack(vis_fwhm_freq).flatten()
mot_fwhm_freq=np.hstack(mot_fwhm_freq).flatten()

#Putting all eggs in one basket
group_metrics = {"vis": {  
          "waveform": vis_wf,
          "burst_rate_av": brate_mean_vis,
          "burst_rate": vis_burst_rate,
          "burst_rate_sem": brate_sem_vis,
          "burst_rate_time": vis_time_plot,
          "peak_freq": vis_peak_freq,
          "peak_amp": vis_peak_amp,
          "fwhm_freq": vis_fwhm_freq,
          "fwhm_time": vis_fwhm_time,          
      },
      "mot": {
          "waveform": mot_wf,
          "burst_rate_av": brate_mean_mot,
          "burst_rate": mot_burst_rate,
          "burst_rate_sem": brate_sem_mot,
          "burst_rate_time": mot_time_plot,
          "peak_freq": mot_peak_freq,
          "peak_amp": mot_peak_amp,
          "fwhm_freq": mot_fwhm_freq,
          "fwhm_time": mot_fwhm_time,
      }
  }



print('Saving results')       
group_metrics_path = op.join(group_path, "SQUID_group_metrics.pkl")
pickle.dump(group_metrics, open(group_metrics_path, "wb"))
print('All done.')
