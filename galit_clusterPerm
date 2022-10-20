import numpy as np

import mne

from scipy.io import loadmat
from itertools import chain
import csv

chanFile = 'C:\\Users\\Malte\\Downloads\\chanList_TFredo.txt'

with open(chanFile, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    ch_names = list(chain.from_iterable((list(reader))))

## Load the data and get an idea of the raw files

# Import tf results
# Set some lists and variables for the loop
conditions = ['AAp', 'PAp', 'AAv', 'PAv']
pType = 'POWtot'
path = 'J:\\RFT Analysis\\TF Data\\'

samples = 600
chNr = 27

my_array=np.zeros(shape=(24, samples, 60, chNr))
c = 0

X = []

# Load every time frequency result exported from matlab and transform it into an mne tfr object
for condName in conditions:

    c = 0

    for cl in ch_names:

        file = path + 'POW_' + condName + '_CSon_all_B2' + '_' + cl + '.mat'
        data = loadmat(file)[pType][:,:,2001:2601]
        data = np.moveaxis(data, [0, 1, 2], [0, -1, 1])

        my_array[:, :, :, c] = data
        c = c + 1

        print('Done with channel ' + cl + ', condition ' + condName)

    print('Done with condition ' + condName)
    X.append(my_array.copy())

# Read info
montage = mne.channels.make_standard_montage(kind='standard_1005')
epochs = mne.read_evokeds('D:\\MEGEEG_project\\MEG\\evoked\\reward64-ave.fif')[0]
epochs.set_montage(montage=montage)
epochs = epochs.pick_channels(ch_names)

import scipy.stats

import mne
from mne.stats import spatio_temporal_cluster_test, combine_adjacency
from mne.channels import find_ch_adjacency

# We are running an F test, so we look at the upper tail
# see also: https://stats.stackexchange.com/a/73993
tail = 1

# We want to set a critical test statistic (here: F), to determine when
# clusters are being formed. Using Scipy's percent point function of the F
# distribution, we can conveniently select a threshold that corresponds to
# some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.001

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = 4
n_observations = 24
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

adjacency, _ = find_ch_adjacency(epochs.info, 'eeg')

# run the cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=f_thresh, tail=tail,
                                             n_jobs=2, adjacency=None)
F_obs, clusters, p_values, _ = cluster_stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# We subselect clusters that we consider significant at an arbitrarily
# picked alpha level: "p_accept".
# NOTE: remember the caveats with respect to "significant" clusters that
# we mentioned in the introduction of this tutorial!
p_accept = 0.52
good_cluster_inds = np.where(p_values < p_accept)[0]

# configure variables for visualization
colors = {"Aud": "crimson", "Vis": 'steelblue'}
linestyles = {"L": '-', "R": '--'}

freqs = [0, 60]
times = [0, 600]

F_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    freq_inds, time_inds, space_inds = clusters[clu_idx]
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    freq_inds = np.unique(freq_inds)

    # get topography for F stat
    f_map = F_obs[freq_inds].mean(axis=0)
    f_map = f_map[time_inds].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vmin=np.min, vmax=np.max, show=False, colorbar=False,
                          mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # add new axis for spectrogram
    ax_spec = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} spectrogram'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += " (max over channels)"
    F_obs_plot = F_obs[..., ch_inds].max(axis=-1)
    F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
    F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
        F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]

    for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ['gray', 'autumn']):
        c = ax_spec.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                           extent=[epochs.times[0], epochs.times[-1],
                                   freqs[0], freqs[-1]])
    ax_spec.set_xlabel('Time (ms)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(title)

    # add another colorbar
    ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(c, cax=ax_colorbar2)
    ax_colorbar2.set_ylabel('F-stat')

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
