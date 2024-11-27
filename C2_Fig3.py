
import pickle
import numpy as np
import os.path as op
import matplotlib.pylab as plt
import mne
from sklearn.preprocessing import RobustScaler

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'


#SQUID: clean waveforms (upon which the PCA was fit)
wf_path = open(op.join(group_path, "SQUID_waveforms_clean.pkl"),'rb')
SQUID_waveforms_clean=pickle.load(wf_path)

#####################
## CLEAN WAVEFORMS ##
#####################

#Plot clean waveforms (those selected for PCA)
burst_times = np.linspace(-0.13, 0.13, num=np.shape(SQUID_waveforms_clean)[1])
mean_waveform = np.mean(SQUID_waveforms_clean, axis=0)
norm_waveform = mean_waveform / mean_waveform.max()

#OPM: clean waveforms (upon which the PCA was fit)
wf_path = open(op.join(group_path, "OPM_waveforms_clean.pkl"),'rb')
OPM_waveforms_clean=pickle.load(wf_path)

#Plot clean waveforms (those selected for PCA)
burst_times = np.linspace(-0.13, 0.13, num=np.shape(OPM_waveforms_clean['x'])[1])
mean_waveform_x = np.mean(OPM_waveforms_clean['x'], axis=0)
mean_waveform_y = np.mean(OPM_waveforms_clean['y'], axis=0)
norm_waveform_x = mean_waveform_x / mean_waveform_x.max()
norm_waveform_y = mean_waveform_y / mean_waveform_y.max()


#Combined waveform plot (SQUID + OPM XY)
f, ax = plt.subplots(1, 3, figsize=(8,6), facecolor="white", dpi=120)
ax[0].plot(burst_times, SQUID_waveforms_clean[:1000,:].T / 1e-15, rasterized=True, lw=0.5, alpha=0.2)
ax[0].plot(burst_times, mean_waveform / 1e-15, lw=2, color="black")
ax[0].set_xlim(burst_times[0], burst_times[-1])
ax[0].set_ylim(-400, 400)
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("fT")
ax[0].set_title('SQUID')
ax[1].plot(burst_times, OPM_waveforms_clean['y'][:1000,:].T / 1e-15, rasterized=True, lw=0.5, alpha=0.2)
ax[1].plot(burst_times, mean_waveform_y / 1e-15, lw=2, color="black")
ax[1].set_xlim(burst_times[0], burst_times[-1])
ax[1].set_ylim(-3000, 3000)
ax[1].set_xlabel("Time [s]")
ax[1].set_title('OPM Radial')
ax[2].plot(burst_times, OPM_waveforms_clean['x'][:1000,:].T / 1e-15, rasterized=True, lw=0.5, alpha=0.2)
ax[2].plot(burst_times, mean_waveform_x / 1e-15, lw=2, color="black")
ax[2].set_xlim(burst_times[0], burst_times[-1])
ax[2].set_ylim(-3000, 3000)
ax[2].set_xlabel("Time [s]")
ax[2].set_title('OPM Tangential')
f.set_figheight(7)
f.set_figwidth(12)  
plt.savefig(op.join(group_path,'Fig3A_Combined_waveforms.png'))
plt.savefig(op.join(group_path,'Fig3A_Combined_waveforms.pdf'))


#############
## Metrics ##
#############

#SQUID
print('SQUID: Loading data')
metrics_file = open(op.join(group_path, "SQUID_group_metrics.pkl"),'rb')
SQUID_metrics=pickle.load(metrics_file)

## Histograms
f, ax = plt.subplots(2, 2, figsize=(6, 7), dpi=100)
cw_m = "#4634eb"
cw_v = "#eb346b"

n_bins=100

#burst duration
val_max=np.ceil(np.max(np.hstack([[SQUID_metrics["vis"]["fwhm_time"]*1e3],[SQUID_metrics["vis"]["fwhm_time"]*1e3]])))
val_min=np.floor(np.min(np.hstack([[SQUID_metrics["vis"]["fwhm_time"]*1e3],[SQUID_metrics["vis"]["fwhm_time"]*1e3]])))
bins=np.linspace(val_min,val_max,n_bins)
ax[0,0].hist(SQUID_metrics["vis"]["fwhm_time"]*1e3, bins=bins, alpha=0.5, color=cw_v);
ax[0,0].hist(SQUID_metrics["mot"]["fwhm_time"]*1e3, bins=bins, alpha=0.5, color=cw_m);
ax[0,0].set_xlim(0, 800)
ax[0,0].set_ylabel("Number of bursts")
ax[0,0].set_xlabel("Duration [ms]")
plt.tight_layout()
for key, spine in ax[0,0].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)
print('SQUID av burst duration VIS: ' +str(np.round(np.mean(SQUID_metrics["vis"]["fwhm_time"]*1e3),1))+ ' ms')
print('SQUID av burst duration MOT: ' +str(np.round(np.mean(SQUID_metrics["mot"]["fwhm_time"]*1e3),1))+ ' ms')


#burst amplitude
n_bins=100
val_max=np.ceil(np.max(np.hstack([[SQUID_metrics["vis"]["peak_amp"]/1e-15],[SQUID_metrics["vis"]["peak_amp"]/1e-15]])))
val_min=np.floor(np.min(np.hstack([[SQUID_metrics["vis"]["peak_amp"]/1e-15],[SQUID_metrics["vis"]["peak_amp"]/1e-15]])))
bins=np.linspace(val_min,val_max,n_bins)
ax[0,1].hist(SQUID_metrics["vis"]['peak_amp']/1e-15, bins=bins, alpha=0.5, color=cw_v)
ax[0,1].hist(SQUID_metrics["mot"]['peak_amp']/1e-15, bins=bins, alpha=0.5, color=cw_m)
ax[0,1].set_xlim(0, 150)
ax[0,1].set_ylabel("Number of bursts")
ax[0,1].set_xlabel("Peak amplitude [fT]")
ax[0,1].ticklabel_format(axis="y", style="sci", scilimits=(10, 3))
plt.tight_layout()
for key, spine in ax[0,1].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)
print('SQUID av Peak amplitude VIS: ' +str(np.round(np.mean(SQUID_metrics["vis"]["peak_amp"]/1e-15),1))+ ' fT')
print('SQUID av Peak amplitude MOT: ' +str(np.round(np.mean(SQUID_metrics["mot"]["peak_amp"]/1e-15),1))+ ' fT')

#Peak frequency
n_bins=19
val_max=np.ceil(np.max(np.hstack([[SQUID_metrics["vis"]["peak_freq"]],[SQUID_metrics["vis"]["peak_freq"]]])))
val_min=np.floor(np.min(np.hstack([[SQUID_metrics["vis"]["peak_freq"]],[SQUID_metrics["vis"]["peak_freq"]]])))
bins=np.arange(val_min,val_max+2,1)
ax[1,0].hist(SQUID_metrics["vis"]["peak_freq"], bins=bins, alpha=0.5, color=cw_v);
ax[1,0].hist(SQUID_metrics["mot"]["peak_freq"], bins=bins, alpha=0.5, color=cw_m);
ax[1,0].set_xlim(13, 30)
ax[1,0].set_ylabel("Number of bursts")
ax[1,0].set_xlabel("Peak Frequency [Hz]")
ax[1,0].ticklabel_format(axis="y", style="sci", scilimits=(10, 3))
plt.tight_layout()
for key, spine in ax[1,0].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)
print('SQUID av Peak frequency VIS: ' +str(np.round(np.mean(SQUID_metrics["vis"]["peak_freq"]),2))+ ' Hz')
print('SQUID av Peak frequency MOT: ' +str(np.round(np.mean(SQUID_metrics["mot"]["peak_freq"]),2))+ ' Hz')

#Frequency span
n_bins=17
val_max=np.ceil(np.max(np.hstack([[SQUID_metrics["vis"]["fwhm_freq"]],[SQUID_metrics["vis"]["fwhm_freq"]]])))
val_min=np.floor(np.min(np.hstack([[SQUID_metrics["vis"]["fwhm_freq"]],[SQUID_metrics["vis"]["fwhm_freq"]]])))
bins=np.arange(val_min,val_max+2,2)
ax[1,1].hist(SQUID_metrics["vis"]["fwhm_freq"], bins=bins, alpha=0.5, color=cw_v);
ax[1,1].hist(SQUID_metrics["mot"]["fwhm_freq"], bins=bins, alpha=0.5, color=cw_m);
ax[1,1].set_xlim(0, 12)
ax[1,1].set_ylabel("Number of bursts")
ax[1,1].set_xlabel("Frequency Span [Hz]")
plt.tight_layout()
for key, spine in ax[1,1].spines.items():
    if key in ["top", "right"]:
        spine.set_visible(False)
print('SQUID av Frequency span VIS: ' +str(np.round(np.mean(SQUID_metrics["vis"]["fwhm_freq"]),2))+ ' Hz')
print('SQUID av Frequency span MOT: ' +str(np.round(np.mean(SQUID_metrics["mot"]["fwhm_freq"]),2))+ ' Hz')

plt.savefig(op.join(group_path,'Fig3B_SQUID_metric_hist.pdf'))
plt.savefig(op.join(group_path,'Fig3B_SQUID_metric_hist.png'))


#OPM
print('OPM: Loading metrics')
opm_metrics_file = open(op.join(group_path, "OPM_group_metrics.pkl"),'rb')
OPM_metrics=pickle.load(opm_metrics_file)

## Histograms
cw_m = "#4634eb"
cw_v = "#eb346b"

for d in ['x','y']:
    
    f, ax = plt.subplots(2, 2, figsize=(6, 7), dpi=100)
    
    #burst duration
    n_bins=100
    val_max=np.ceil(np.max(np.hstack([[OPM_metrics[d]["vis"]["fwhm_time"]*1e3],[OPM_metrics[d]["vis"]["fwhm_time"]*1e3]])))
    val_min=np.floor(np.min(np.hstack([[OPM_metrics[d]["vis"]["fwhm_time"]*1e3],[OPM_metrics[d]["vis"]["fwhm_time"]*1e3]])))
    bins=np.linspace(val_min,val_max,n_bins)
    ax[0,0].hist(OPM_metrics[d]["vis"]["fwhm_time"]*1e3, bins=100, alpha=0.5, color=cw_v);
    ax[0,0].hist(OPM_metrics[d]["mot"]["fwhm_time"]*1e3, bins=100, alpha=0.5, color=cw_m);
    ax[0,0].set_xlim(0, 800)
    ax[0,0].set_ylabel("Number of bursts")
    ax[0,0].set_xlabel("Duration [ms]")
    ax[0,0].set_title('Burst Duration')
    plt.tight_layout()
    print('OPM ' +d+ ' av burst duration VIS: ' +str(np.round(np.mean(OPM_metrics[d]["vis"]["fwhm_time"]*1e3),1))+ ' ms')
    print('OPM ' +d+ ' av burst duration MOT: ' +str(np.round(np.mean(OPM_metrics[d]["mot"]["fwhm_time"]*1e3),1))+ ' ms')
    
    #burst amplitude
    n_bins=100
    val_max=np.ceil(np.max(np.hstack([[OPM_metrics[d]["vis"]["peak_amp"]/1e-15],[OPM_metrics[d]["vis"]["peak_amp"]/1e-15]])))
    val_min=np.floor(np.min(np.hstack([[OPM_metrics[d]["vis"]["peak_amp"]/1e-15],[OPM_metrics[d]["vis"]["peak_amp"]/1e-15]])))
    bins=np.linspace(val_min,val_max,n_bins)
    ax[0,1].hist(OPM_metrics[d]["vis"]['peak_amp']/1e-15, bins=bins, alpha=0.5, color=cw_v)
    ax[0,1].hist(OPM_metrics[d]["mot"]['peak_amp']/1e-15, bins=bins, alpha=0.5, color=cw_m)
    ax[0,1].set_xlim(0, 1000)
    ax[0,1].set_ylabel("Number of bursts")
    ax[0,1].set_xlabel("Peak amplitude [fT]")
    ax[0,1].ticklabel_format(axis="y", style="sci", scilimits=(10, 3))
    plt.tight_layout()
    print('OPM ' +d+ ' av Peak amplitude VIS: ' +str(np.round(np.mean(OPM_metrics[d]["vis"]["peak_amp"]/1e-15),1))+ ' fT')
    print('OPM ' +d+ ' av Peak amplitude MOT: ' +str(np.round(np.mean(OPM_metrics[d]["mot"]["peak_amp"]/1e-15),1))+ ' fT')
    
    #Peak frequency
    n_bins=17
    val_max=np.ceil(np.max(np.hstack([[OPM_metrics[d]["vis"]["peak_freq"]],[OPM_metrics[d]["vis"]["peak_freq"]]])))
    val_min=np.floor(np.min(np.hstack([[OPM_metrics[d]["vis"]["peak_freq"]],[OPM_metrics[d]["vis"]["peak_freq"]]])))    
    bins=np.arange(val_min,val_max+2,1)
    ax[1,0].hist(OPM_metrics[d]["vis"]["peak_freq"], bins=bins, alpha=0.5, color=cw_v);
    ax[1,0].hist(OPM_metrics[d]["mot"]["peak_freq"], bins=bins, alpha=0.5, color=cw_m);
    ax[1,0].set_xlim(16, 30)
    ax[1,0].set_ylim(0,1.2e4)
    ax[1,0].set_ylabel("Number of bursts")
    ax[1,0].set_xlabel("Peak Frequency [Hz]")
    ax[1,0].ticklabel_format(axis="y", style="sci", scilimits=(10, 3))
    plt.tight_layout()
    print('OPM ' +d+ ' av Peak frequency VIS: ' +str(np.round(np.mean(OPM_metrics[d]["vis"]["peak_freq"]),2))+ ' Hz')
    print('OPM ' +d+ ' av Peak frequency MOT: ' +str(np.round(np.mean(OPM_metrics[d]["mot"]["peak_freq"]),2))+ ' Hz')
    
    #Frequency span
    n_bins=17
    val_max=np.ceil(np.max(np.hstack([[OPM_metrics[d]["vis"]["fwhm_freq"]],[OPM_metrics[d]["vis"]["fwhm_freq"]]])))
    val_min=np.floor(np.min(np.hstack([[OPM_metrics[d]["vis"]["fwhm_freq"]],[OPM_metrics[d]["vis"]["fwhm_freq"]]])))
    bins=np.arange(val_min,val_max+2,2)
    ax[1,1].hist(OPM_metrics[d]["vis"]["fwhm_freq"], bins=bins, alpha=0.5, color=cw_v);
    ax[1,1].hist(OPM_metrics[d]["mot"]["fwhm_freq"], bins=bins, alpha=0.5, color=cw_m);
    ax[1,1].set_xlim(0, 12)
    ax[1,1].set_ylabel("Number of bursts")
    ax[1,1].set_xlabel("Frequency Span [Hz]")
    plt.tight_layout()    
    print('OPM ' +d+ ' av Frequency span VIS: ' +str(np.round(np.mean(OPM_metrics[d]["vis"]["fwhm_freq"]),2))+ ' Hz')
    print('OPM ' +d+ ' av Frequency span MOT: ' +str(np.round(np.mean(OPM_metrics[d]["mot"]["fwhm_freq"]),2))+ ' Hz')
    
    plt.savefig(op.join(group_path,'Fig3C_OPM' +d+ '_metric_hist.pdf'))
    plt.savefig(op.join(group_path,'Fig3C_OPM' +d+ '_metric_hist.png'))

# PCA SELECTION
PCA_file = open(op.join(group_path, "comb_PCA_allnoZ.pkl"),'rb')
PCA_saved=pickle.load(PCA_file)

metrics_file = open(op.join(group_path, "SQUID_sub_metrics.pkl"),'rb')
metrics=pickle.load(metrics_file)

PC_var_exp = PCA_saved.explained_variance_ratio_
PC_r = np.arange(PC_var_exp.shape[0]) + 1
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_MEG-ave.fif'))
vis_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_MEG-ave.fif'))
mot_times=tmp[0].times
buffer = 0.125
bin_width=0.05
baseline_range = [-0.5, -0.25]
visual_time_bins = np.arange(vis_times[0] + buffer, vis_times[-1] - buffer, bin_width)
motor_time_bins = np.arange(mot_times[0] + buffer, mot_times[-1] - buffer, bin_width)
visual_bin_ranges = list(zip(visual_time_bins[:-1], visual_time_bins[1:]))
motor_bin_ranges = list(zip(motor_time_bins[:-1], motor_time_bins[1:]))

scaler = RobustScaler().fit(SQUID_waveforms_clean)
waveforms_scaled = scaler.transform(SQUID_waveforms_clean)

PC_score_in_time = {sub: {
   ep: {
      "PC_{}".format(pc+1) : [] for pc in range(20)
   } for ep in ["mot", "vis"] 
} for sub in metrics.keys()}

for sub in metrics.keys():
        for ep in metrics[sub].keys():
            wvf = scaler.transform(metrics[sub][ep]["waveform"])
            metrics[sub][ep]["PC_score"] = PCA_saved.transform(wvf)
            for pc_ix in range(20):
                pc_key = "PC_{}".format(pc_ix+1)                
                if ep == "vis":
                    bin_ranges = visual_bin_ranges
                elif ep == "mot":
                    bin_ranges = motor_bin_ranges
                tri_tc = []
                for st, en in bin_ranges:
                    tr_ix = np.where(
                        (metrics[sub][ep]["peak_time"] >= st) &
                        (metrics[sub][ep]["peak_time"] <= en)                       
                    )[1]
                    res = np.nanmean(metrics[sub][ep]["PC_score"][tr_ix, pc_ix])
                    tri_tc.append(res)
                tri_tc = np.array(tri_tc)
                PC_score_in_time[sub][ep][pc_key].append(tri_tc)

PC_mean_score_in_time = {
       ep: {
          "PC_{}".format(pc+1) : [] for pc in range(20)
       } for ep in ["mot", "vis"] 
    }

for sub in PC_score_in_time.keys():
    for ep in PC_score_in_time[sub].keys():
        for pc_key in PC_score_in_time[sub][ep].keys():
            data = np.vstack(PC_score_in_time[sub][ep][pc_key])
            data = np.nanmean(data, axis=0)
            PC_mean_score_in_time[ep][pc_key].append(data)

PC_fit_score = {ep: [] for ep in ["mot", "vis"]}
for ep in PC_mean_score_in_time.keys():
    for pc_ix in PC_mean_score_in_time[ep].keys():
        data = np.vstack(PC_mean_score_in_time[ep][pc_ix])                
        x= np.arange(np.shape(data)[1])        
        m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, np.mean(data,axis=0), rcond=None)[0]        
        PC_fit_score[ep].append(m)

SQUID_fit_sc=np.array(PC_fit_score['vis'])*-1
SQUID_fit_sc[SQUID_fit_sc<0]=0

#get OPM fit_score
print('Loading subject metrics')
metrics_file = open(op.join(group_path, "OPM_sub_metrics.pkl"),'rb')
metrics=pickle.load(metrics_file)
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_VIS_OPM-ave.fif'))
vis_times=tmp[0].times
tmp=mne.read_evokeds(op.join(group_path,'evoked','01_MOT_OPM-ave.fif'))
mot_times=tmp[0].times
buffer = 0.125
bin_width=0.05
baseline_range = [-0.5, -0.25]
visual_time_bins = np.arange(vis_times[0] + buffer, vis_times[-1] - buffer, bin_width)
motor_time_bins = np.arange(mot_times[0] + buffer, mot_times[-1] - buffer, bin_width)
visual_bin_ranges = list(zip(visual_time_bins[:-1], visual_time_bins[1:]))
motor_bin_ranges = list(zip(motor_time_bins[:-1], motor_time_bins[1:]))
print('Loading data')
wf_file = open(op.join(group_path, "OPM_waveforms_clean.pkl"),'rb')
wf_clean=pickle.load(wf_file)

OPMscaler={k : [] for k in wf_clean.keys()}
OPMscaler['x'] = RobustScaler().fit(wf_clean['x'])
OPMscaler['y'] = RobustScaler().fit(wf_clean['y'])

PC_score_in_time = {k : {sub: { ep: 
                               {"PC_{}".format(pc+1) : [] for pc in range(20)}
                               for ep in ["mot", "vis"] } 
                               for sub in metrics[k].keys() } 
                               for k in ['x','y']}    
for i in ['x','y']:
    for sub in metrics[i].keys():
        for ep in metrics[i][sub].keys():
            wvf = OPMscaler[i].transform(metrics[i][sub][ep]["waveform"])
            metrics[i][sub][ep]["PC_score"] = PCA_saved.transform(wvf)
            for pc_ix in range(20):
                pc_key = "PC_{}".format(pc_ix+1)                
                if ep == "vis":
                    bin_ranges = visual_bin_ranges
                elif ep == "mot":
                    bin_ranges = motor_bin_ranges
                tri_tc = []
                for st, en in bin_ranges:
                    tr_ix = np.where(
                        (metrics[i][sub][ep]["peak_time"] >= st) &
                        (metrics[i][sub][ep]["peak_time"] <= en)                       
                    )[1]
                    res = np.nanmean(metrics[i][sub][ep]["PC_score"][tr_ix, pc_ix])
                    tri_tc.append(res)
                tri_tc = np.array(tri_tc)
                PC_score_in_time[i][sub][ep][pc_key].append(tri_tc)


PC_mean_score_in_time = {k : { ep: 
                              {"PC_{}".format(pc+1) : [] for pc in range(20)} 
                              for ep in ["mot", "vis"] }
                              for k in ['x','y']} 
    
for i in PC_score_in_time.keys():
    for sub in PC_score_in_time[i].keys():
        for ep in PC_score_in_time[i][sub].keys():
            for pc_key in PC_score_in_time[i][sub][ep].keys():
                data = np.vstack(PC_score_in_time[i][sub][ep][pc_key])
                data = np.nanmean(data, axis=0)
                PC_mean_score_in_time[i][ep][pc_key].append(data)

PC_mean_variance_of_score_in_time = {k : {ep: [] for ep in ["mot", "vis"]} for k in metrics.keys()}

for i in PC_mean_score_in_time.keys():
    for ep in PC_mean_score_in_time[i].keys():
        for pc_ix in PC_mean_score_in_time[i][ep].keys():
            data = np.vstack(PC_mean_score_in_time[i][ep][pc_ix])
            data = np.var(np.mean(data, axis=0))
            PC_mean_variance_of_score_in_time[i][ep].append(data)

PC_fit_score = {axs: {ep: [] for ep in ["mot", "vis"]} for axs in PC_mean_score_in_time.keys()}
for axs in PC_mean_score_in_time.keys():
    for ep in PC_mean_score_in_time[axs].keys():
        for pc_ix in PC_mean_score_in_time[axs][ep].keys():
            data = np.vstack(PC_mean_score_in_time[axs][ep][pc_ix])        
            x= np.arange(np.shape(data)[1])        
            m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, np.mean(data,axis=0), rcond=None)[0]            
            PC_fit_score[axs][ep].append(m)                       


OPMX_fit_sc=np.array(PC_fit_score['x']['vis'])*-1
OPMY_fit_sc=np.array(PC_fit_score['y']['vis'])*-1

#plot using selected overlapping components
f, ax = plt.subplots(1, 1, figsize=(10,3.5), dpi=300, facecolor="white")
comps_selected=np.zeros(20)
comps_selected[5:8]=1
bar = ax.bar(PC_r, comps_selected, lw=0.2, color="black", alpha=0.2, width=1)
bar = ax.bar(PC_r, PC_var_exp, lw=0.2)
cm = plt.cm.get_cmap("turbo_r")
for i in PC_r:
    plt.setp(bar[i-1],"facecolor", cm(i/20))

ax.set_ylim(0, 0.15)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Proportion of \nVariance Explained")
ax.set_xticks(PC_r)

ax2 = ax.twinx()
ax2.scatter(PC_r, SQUID_fit_sc, color="blue", zorder=10)
ax2.plot(PC_r, SQUID_fit_sc, color="blue", zorder=10)
ax2.scatter(PC_r, OPMY_fit_sc, color="red", zorder=10)
ax2.plot(PC_r, OPMY_fit_sc, color="red", zorder=10)
ax2.scatter(PC_r, OPMX_fit_sc, color="green", zorder=10)
ax2.plot(PC_r, OPMX_fit_sc, color="green", zorder=10)
ax2.set_ylabel("Component score", color="blue")
ax2.set_ylim(0, 0.016)
plt.tight_layout();

plt.savefig(op.join(group_path,'Fig3D_SQUID_PCA_histogram_OverlapComp.pdf'))
plt.savefig(op.join(group_path,'Fig3D_SQUID_PCA_histogram_OverlapComp.png'))
