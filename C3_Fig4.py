import pickle
import os.path as op
import numpy as np
import sys
from matplotlib import colors
from sklearn.preprocessing import RobustScaler
import mne
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter
from mne.stats import permutation_cluster_1samp_test

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/Motor/'
sys.path.append(group_path)

print('Loading data')
wf_file = open(op.join(group_path, "SQUID_waveforms_clean.pkl"),'rb')
wf_clean=pickle.load(wf_file)

PCA_file = open(op.join(group_path, "comb_PCA_allnoZ.pkl"),'rb')
PCA_saved=pickle.load(PCA_file)

scaler = RobustScaler().fit(wf_clean)
waveforms_scaled = scaler.transform(wf_clean)

print('SQUID: Loading metrics')
metrics_file = open(op.join(group_path, "SQUID_sub_metrics.pkl"),'rb')
metrics=pickle.load(metrics_file)

print('SQUID: Loading data')
metrics_file = open(op.join(group_path, "SQUID_group_metrics.pkl"),'rb')
group_metrics=pickle.load(metrics_file)

#Variance over time
#We need to know the trial timecourse, so load an evoked for both conditions and extract times
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


#a little bit of prep - create waveform array with all waveforms, and 
pc_list = ["PC_{}".format(pc+1) for pc in range(20)]
imp_features = ['peak_time', 'peak_freq', 'peak_amp_base', 'fwhm_freq', 'fwhm_time', 'trial', 'pp_ix']
waveform_array = []

data_frame_dict = {    
    i: [] for i in imp_features + pc_list
}

for sub in metrics.keys():
    for ep in metrics[sub].keys():        
        for ft in imp_features:
            if ft == 'pp_ix':
                data_frame_dict[ft].append(metrics[sub][ep][ft])
            else:
                data_frame_dict[ft].extend(metrics[sub][ep][ft])
    
        
        for pc_ix, pc_key in enumerate(pc_list):
            data_frame_dict[pc_key].extend(metrics[sub][ep]["PC_score"][:, pc_ix])
        
        waveform_array.append(metrics[sub][ep]["waveform"])

waveform_array = np.vstack(waveform_array)
burst_features= data_frame_dict

#Modify waveform array
pc_keys = ["PC_{}".format(pc_ix+1) for pc_ix in range(20)]
prct = np.linspace(0,100, num=11)
prct_ranges = list(zip(prct[:-1], prct[1:]))

wvfrms = {
    k: [] for k in pc_keys
}

for pc_ix, pc_key in enumerate(pc_keys):
    for low, hi in prct_ranges:
        low_perc = np.percentile(burst_features[pc_key], low)
        hi_perc = np.percentile(burst_features[pc_key], hi)
        wvf_ixs=np.where((burst_features[pc_key]>=low_perc) & (burst_features[pc_key]<=hi_perc))        
        MWF = np.mean(waveform_array[wvf_ixs, :], axis=1)
        wvfrms[pc_key].append(MWF)


#Plot componernt shape and timecourse
waveform_time = np.linspace(-0.13, 0.13, num=np.shape(waveform_array)[1])
col_r = plt.cm.cool(np.linspace(0,1, num=10))

mean_waveform = (np.mean(waveform_array, axis=0))
vis_time_plot = visual_time_bins[:-1]
mot_time_plot = motor_time_bins[:-1]

  
PC_to_analyse=['PC_6', 'PC_7', 'PC_8']
PC_burst_rate_spec = {i: {"vis": [], "mot": []} for i in PC_to_analyse}

subjects = np.unique(burst_features['pp_ix'])

time_bins = {
    "vis": visual_time_bins,
    "mot": motor_time_bins
}

for pc_key in PC_to_analyse:
    pc_ind=pc_list.index(pc_key)
    comp_score = burst_features[pc_key]
    score_range = np.linspace(
        np.percentile(comp_score, 0.5),
        np.percentile(comp_score, 99.5),
        num = 41
    )
    for sub in subjects:
        sub_PC_br = {
            "vis": [],
            "mot": []
        }
        for ep in ["vis", "mot"]:
            peak_times=metrics[sub][ep]['peak_time'][0]
            pc_scores=metrics[sub][ep]['PC_score'][:,pc_ind]

            PC_br, t_bin, m_bin = np.histogram2d(
                peak_times,
                pc_scores,
                bins = [time_bins[ep], score_range]
            )
            PC_br = PC_br / bin_width
            PC_br = gaussian_filter(PC_br, [1,1])            
            sub_PC_br[ep] = PC_br
    
        for ep in ["vis", "mot"]:
            PC_burst_rate_spec[pc_key][ep].append(sub_PC_br[ep])
        
#Modify waveform array
pc_keys = ["PC_{}".format(pc_ix+1) for pc_ix in range(20)]
prct = np.linspace(0,100, num=5)
prct_ranges = list(zip(prct[:-1], prct[1:]))

wvfrms = {
    k: [] for k in pc_keys
}

for pc_ix, pc_key in enumerate(pc_keys):
    for low, hi in prct_ranges:
        low_perc = np.percentile(burst_features[pc_key], low)
        hi_perc = np.percentile(burst_features[pc_key], hi)
        wvf_ixs=np.where((burst_features[pc_key]>=low_perc) & (burst_features[pc_key]<=hi_perc))        
        MWF = np.mean(waveform_array[wvf_ixs, :], axis=1)
        wvfrms[pc_key].append(MWF)        


############################
#burst rate component plot #
############################
PC_to_analyse=['PC_6', 'PC_7', 'PC_8']

n_permutations=1000
p=0.05

vis_xticks = [-0.5, 0, 0.5, 1, 1.5]
mot_xticks = [-0.5, 0, 0.5, 1]

score_prc_range = np.linspace(0, 100, 40)
waveform_time = np.linspace(-0.13, 0.13, num=np.shape(waveform_array)[1])
col_r = plt.cm.cool(np.linspace(0,1, num=4))
graynorm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=1.1)

prct = np.linspace(0,100, num=5)
prct_ranges = list(zip(prct[:-1], prct[1:]))

f, ax = plt.subplots(
    len(PC_to_analyse), 5, figsize=(19, len(PC_to_analyse)*3+3), 
    facecolor="white", gridspec_kw={'width_ratios': [1, 2.15, 1.65, 2.15, 1.65]},
    constrained_layout=True
)

#calculate significant for average burst rate
sig_lines_vis_av=[]
selection=np.array(group_metrics['vis']['burst_rate'])

df=np.shape(selection)[0]-1
t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
T_obs_vis, clusters_vis, cluster_p_values_vis, H0_vis = permutation_cluster_1samp_test(
                                        selection, 
                                        n_permutations=n_permutations,
                                        threshold=t_thresh,
                                        tail=0,
                                        adjacency=None,
                                        out_type='mask',
                                        verbose=True)
#save significance
for i_c, cl in enumerate(clusters_vis):
    if cluster_p_values_vis[i_c] <= 0.05:
        sig=vis_time_plot[cl] 
        sig_lines_vis_av.append([sig[0],sig[-1]])
        
sig_lines_mot_av=[]
selection=np.array(group_metrics['mot']['burst_rate'])

df=np.shape(selection)[0]-1
t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
T_obs_mot, clusters_mot, cluster_p_values_mot, H0_mot = permutation_cluster_1samp_test(
                                        selection, 
                                        n_permutations=n_permutations,
                                        threshold=t_thresh,
                                        tail=0,
                                        adjacency=None,
                                        out_type='mask',
                                        verbose=True)
#save significance
for i_c, cl in enumerate(clusters_mot):
    if cluster_p_values_mot[i_c] <= 0.05:
        sig=mot_time_plot[cl] 
        sig_lines_mot_av.append([sig[0],sig[-1]])        
        
for pc_ix, pc_key in enumerate(PC_to_analyse):  
    label = " ".join([i for i in pc_key.split("_")])
    ax[pc_ix, 0].set_title(label)
    for key, spine in ax[pc_ix, 0].spines.items():
        spine.set_visible(False)
    ax[pc_ix, 0].set_yticks([])
    ax[pc_ix, 0].set_xticks([])
    for pr in range(4):
        ax[pc_ix, 0].plot(waveform_time, wvfrms[pc_key][pr].T*1e14 + np.mean([prct[pr], prct[pr+1]]), c=col_r[pr])
    ax[pc_ix, 0].plot(waveform_time, mean_waveform*1e14 + 50, c="black", lw=1, linestyle="dotted")
    
    # Time-burst rate
    selection_vis=np.mean(PC_burst_rate_spec[pc_key]["vis"], axis=0).T
    selection_vis_all=PC_burst_rate_spec[pc_key]["vis"].copy()
    selection_mot=np.mean(PC_burst_rate_spec[pc_key]["mot"], axis=0).T
    selection_mot_all=PC_burst_rate_spec[pc_key]["mot"].copy()
    # baselining
    bl_ix = np.where(
        (time_bins["vis"] >= baseline_range[0]) &
        (time_bins["vis"] <= baseline_range[-1])
        )[0]
    baseline = np.mean(selection_vis[:,bl_ix], axis=1).reshape(1, -1)
    selection_vis = selection_vis-baseline.T
    selection_mot = selection_mot-baseline.T
    #remove small values        
    selection_vis = selection_vis/baseline.T        
    selection_mot = selection_mot/baseline.T   
    #baseline all
    for s in range(np.shape(selection_vis_all)[0]):
        bl_all=np.nanmean(selection_vis_all[s][bl_ix,],axis=0)
        bl_all[np.isnan(bl_all)]=0 #set nan to zero
        selection_vis_all[s]=selection_vis_all[s]-bl_all
        selection_mot_all[s]=selection_mot_all[s]-bl_all
        bl_all[bl_all==0]=1 #set zero to one
        selection_vis_all[s]=(selection_vis_all[s]/bl_all).T
        selection_mot_all[s]=(selection_mot_all[s]/bl_all).T
     
    #compute spatio-temporal clustering stat
    prct_sel=np.shape(selection_vis_all)[1]
    times_sel=np.shape(selection_vis_all)[2]
    adjacency_vis = mne.stats.combine_adjacency(prct_sel, times_sel)
    
    df=np.shape(selection_vis_all)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_vis, clusters_vis, cluster_p_values_vis, H0_vis = permutation_cluster_1samp_test(
                                            np.asarray(selection_vis_all), 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_vis,
                                            out_type='mask',
                                            verbose=True)
    
    T_obs_plot_vis = np.nan * np.ones_like(T_obs_vis)
    for c, p_val in zip(clusters_vis, cluster_p_values_vis):
        if p_val <= 0.05:
            T_obs_plot_vis[c] = T_obs_vis[c]
            
    prct_sel=np.shape(selection_mot_all)[1]
    times_sel=np.shape(selection_mot_all)[2]
    adjacency_mot = mne.stats.combine_adjacency(prct_sel, times_sel)        
    df=np.shape(selection_mot_all)[0]-1
    t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
    T_obs_mot, clusters_mot, cluster_p_values_mot, H0_vis = permutation_cluster_1samp_test(
                                            np.asarray(selection_mot_all), 
                                            n_permutations=n_permutations,
                                            threshold=t_thresh,
                                            tail=0,
                                            adjacency=adjacency_mot,
                                            out_type='mask',
                                            verbose=True)
    
    T_obs_plot_vis = np.nan * np.ones_like(T_obs_vis)
    for c, p_val in zip(clusters_vis, cluster_p_values_vis):
        if p_val <= 0.05:
            T_obs_plot_vis[c] = T_obs_vis[c]
            
    T_obs_plot_mot = np.nan * np.ones_like(T_obs_mot)
    for c, p_val in zip(clusters_mot, cluster_p_values_mot):
        if p_val <= 0.05:
            T_obs_plot_mot[c] = T_obs_mot[c]

        
    max_abs = np.max([
        np.abs(selection_vis).max(),
        np.abs(selection_mot).max()
    ])
    
    divnorm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    # BURSTxSCORE spectrum vis
    ax[pc_ix, 1].imshow(
        selection_vis,
        origin="lower", aspect="auto", norm=divnorm, cmap="RdBu_r",
        extent=[visual_time_bins[0], visual_time_bins[-1], 0, 100]
    )
    
    # BURSTxSCORE stat vis
    ax[pc_ix, 1].imshow(
        np.isnan(T_obs_plot_vis)*1, origin="lower", aspect="auto",
        norm=graynorm, cmap="gray_r", alpha=0.4,
        extent=[visual_time_bins[0], visual_time_bins[-1], 0, 100]
    )
    ax[pc_ix, 1].set_xticks(vis_xticks)
    ax[pc_ix, 1].set_yticks(prct)
    
    for pr in prct:
        ax[pc_ix, 1].axhline(pr, lw=0.5, linestyle=(0, (5, 10)), c="blue")
    ax[pc_ix, 1].axvline(0, lw=1, linestyle=(0, (5, 10)), c="black")
    
    # BURSTxSCORE spectrum mot
    im_mot = ax[pc_ix, 2].imshow(
        selection_mot,
        origin="lower", aspect="auto", norm=divnorm, cmap="RdBu_r",
        extent=[motor_time_bins[0], motor_time_bins[-1], 0, 100]
    )
    
    # BURSTxSCORE stat mot
    ax[pc_ix, 2].imshow(
        np.isnan(T_obs_plot_mot)*1, origin="lower", aspect="auto",
        norm=graynorm, cmap="gray_r", alpha=0.4,
        extent=[motor_time_bins[0], motor_time_bins[-1], 0, 100]
    )

    ax[pc_ix, 2].set_yticks([])
    ax[pc_ix, 2].set_xticks(mot_xticks)
    cbar = plt.colorbar(im_mot, ax=ax[pc_ix, 2])
    
    for pr in prct:
        ax[pc_ix, 2].axhline(pr, lw=0.5, linestyle=(0, (5, 10)), c="blue")
    ax[pc_ix, 2].axvline(0, lw=1, linestyle=(0, (5, 10)), c="black")
    
    #burst rate time course
    brate_mean_vis=group_metrics['vis']['burst_rate_av']
    brate_sem_vis=group_metrics['vis']['burst_rate_sem']
    
    ax[pc_ix, 3].plot(vis_time_plot, brate_mean_vis, lw=1, color="black")
    ax[pc_ix, 3].fill_between(
        vis_time_plot, 
        brate_mean_vis - brate_sem_vis,
        brate_mean_vis + brate_sem_vis,
        lw=0, color="black", alpha=0.2
    )
    ax[pc_ix, 3].axvline(0, lw=1, linestyle=(0, (5, 10)), c="black")
    
   
    bl_vis=np.zeros((len(prct_ranges),np.shape(PC_burst_rate_spec[pc_key]["vis"])[0]))
    sig_lines_vis=[]
    for c_ix, (b, e) in enumerate(prct_ranges):
        ixes = np.where(((score_prc_range) >= b) & (score_prc_range <= e))[0]
        selection=np.array(PC_burst_rate_spec[pc_key]["vis"])[:,:, ixes]
        selection = np.nanmean(selection, axis=2)        
        # baselining
        bl_ix = np.where(
            (time_bins["vis"] >= baseline_range[0]) &
            (time_bins["vis"] <= baseline_range[-1])
            )[0]
        baseline = np.mean(selection[:,bl_ix], axis=1).reshape(1, -1)
        bl_vis[c_ix,]=baseline
        selection = selection-baseline.T
        #remove small values        
        baseline[baseline < 1]=1        
        selection = selection/baseline.T        
                
        #remove inf, if present
        selection[np.isposinf(selection)]=np.nan
        selection[np.isneginf(selection)]=np.nan
        
        #stats
        df=np.shape(selection)[0]-1
        t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
        T_obs_vis, clusters_vis, cluster_p_values_vis, H0_vis = permutation_cluster_1samp_test(
                                                selection, 
                                                n_permutations=n_permutations,
                                                threshold=t_thresh,
                                                tail=0,
                                                adjacency=None,
                                                out_type='mask',
                                                verbose=True)
        
        mean = np.nanmean(selection, axis=0)
        sem = np.std(selection, axis=0) / np.sqrt(np.shape(selection)[0])
        ax[pc_ix, 3].plot(vis_time_plot, mean, lw=1, color=col_r[c_ix])
        ax[pc_ix, 3].fill_between(
            vis_time_plot, 
            mean - sem,
            mean + sem,
            lw=0, color=col_r[c_ix], alpha=0.2
        )
        
        #save significance
        for i_c, cl in enumerate(clusters_vis):
            if cluster_p_values_vis[i_c] <= 0.05:
                sig=vis_time_plot[cl] 
                sig_lines_vis.append([sig[0],sig[-1],c_ix])
    
    for key, spine in ax[pc_ix, 3].spines.items():
        if key in ["top", "right"]:
            spine.set_visible(False)
    
    brate_mean_mot=group_metrics['mot']['burst_rate_av']
    brate_sem_mot=group_metrics['mot']['burst_rate_sem']
    
    ax[pc_ix, 4].plot(mot_time_plot, brate_mean_mot, lw=1, color="black")
    ax[pc_ix, 4].fill_between(
        mot_time_plot, 
        brate_mean_mot - brate_sem_mot,
        brate_mean_mot + brate_sem_mot,
        lw=0, color="black", alpha=0.2
    )
    ax[pc_ix, 4].axvline(0, lw=1, linestyle=(0, (5, 10)), c="black")
        
    sig_lines_mot=[]
    for c_ix, (b, e) in enumerate(prct_ranges):
        ixes = np.where(((score_prc_range) >= b) & (score_prc_range <= e))[0]
        selection=np.array(PC_burst_rate_spec[pc_key]["mot"])[:,:, ixes]
        selection = np.nanmean(selection, axis=2)        
        baseline =bl_vis[c_ix,]
        selection = selection-baseline.reshape(-1,1)
        #remove small values
        baseline[baseline < 1]=1        
        selection = selection/baseline.reshape(-1,1)
             
        #stats
        df=np.shape(selection)[0]-1
        t_thresh=scipy.stats.t.ppf(1-(p/2),df=df)
        T_obs_mot, clusters_mot, cluster_p_values_mot, H0_mot = permutation_cluster_1samp_test(
                                                selection, 
                                                n_permutations=n_permutations,
                                                threshold=t_thresh,
                                                tail=0,
                                                adjacency=None,
                                                out_type='mask',
                                                verbose=True)
        
        mean = np.mean(selection, axis=0)
        sem = np.std(selection, axis=0) / np.sqrt(np.shape(selection)[0])
        ax[pc_ix, 4].plot(mot_time_plot, mean, lw=1, color=col_r[c_ix])
        ax[pc_ix, 4].fill_between(
            mot_time_plot, 
            mean - sem,
            mean + sem,
            lw=0, color=col_r[c_ix], alpha=0.2
        )
        
        #save significance
        for i_c, cl in enumerate(clusters_mot):
            if cluster_p_values_mot[i_c] <= 0.05:
                sig=mot_time_plot[cl] 
                sig_lines_mot.append([sig[0],sig[-1],c_ix])
    
    ax[pc_ix, 4].set_yticks([])
    for key, spine in ax[pc_ix, 4].spines.items():
        if key in ["top", "left", "right"]:
            spine.set_visible(False)
    
    ymin = np.min([ax[pc_ix, 3].get_ylim()[0], ax[pc_ix, 4].get_ylim()[0]])
    ymax = np.max([ax[pc_ix, 3].get_ylim()[1], ax[pc_ix, 4].get_ylim()[1]])
    
    sig_y=-.5
    
    #plot significance lines
    nlines=np.max([sig_lines_vis[-1][2],sig_lines_mot[-1][2]])
    if len(sig_lines_vis_av)>0:
        nlines=nlines+1
        for sl in sig_lines_vis_av:
            ax[pc_ix, 3].hlines(sig_y,sl[0],sl[1],color='k')    
    
    for sl in sig_lines_vis:
        ax[pc_ix, 3].hlines(sig_y+((nlines-sl[2])*-.05),sl[0],sl[1],color=col_r[sl[2]])
        
    if len(sig_lines_mot_av)>0:
        nlines=nlines+1
        for sl in sig_lines_mot_av:
            ax[pc_ix, 4].hlines(sig_y*-.05,sl[0],sl[1],color='k')
    
    for sl in sig_lines_mot:
        ax[pc_ix, 4].hlines(sig_y+((nlines-sl[2])*-.05),sl[0],sl[1],color=col_r[sl[2]])
    
    
    ymin=-.65
    ymax=.8  
    
    ax[pc_ix, 1].set_ylabel("Percentile of PC score")
    ax[pc_ix, 3].set_ylim(ymin, ymax)
    ax[pc_ix, 3].set_xlim(-.5, vis_time_plot[-1])
    ax[pc_ix, 3].set_ylabel("Burst rate")
    ax[pc_ix, 4].set_ylim(ymin, ymax)
    ax[pc_ix, 4].set_xlim(mot_time_plot[0], mot_time_plot[-1])
    
    

ax[pc_ix, 0].set_ylabel("Burst Shape")
ax[pc_ix, 1].set_xlabel("Time from Visual Stimulus onset [s]")

ax[pc_ix, 2].set_xlabel("Time from Motor response offset [s]")
ax[pc_ix, 3].set_xlabel("Time from Visual Stimulus onset [s]")
ax[pc_ix, 4].set_xlabel("Time from Motor response offset [s]")

plt.savefig(op.join(group_path,'Fig4_SQUID_scaled.pdf'))
plt.savefig(op.join(group_path,'Fig4_SQUID_scaled.png'))
