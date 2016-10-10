from __future__ import division, print_function
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import argparse
from collections import namedtuple
from spiketopics.helpers import frames_to_times, get_movie_partition


# input data matrix from matlab files
datadir = 'data'
fname = 'toroid20120718a.mat'
datfile = os.path.join(datadir, fname)

# keyboard input of # time bins and full dataset process
parser = argparse.ArgumentParser(description="Prepare data and trial times for experiment")
parser.add_argument("integers", type=int, help="input number of time bins")
parser.add_argument("input", help="use full dataset or exclude outliers")

args = parser.parse_args()

# input the time bins
kk = args.integers
# full dataset
if args.input == 'full':
    fulldata = True
else:
    fulldata = False

# load data
datmat = sio.loadmat(datfile)['dat']

# get header
header = datmat['h'][0][0]

# get events (trial x time)
evtdat = datmat['c'][0][0]
evtnames = header['vv'][0][0][0].dtype.names
evt = pd.DataFrame(evtdat, columns=evtnames[:10])
evt = evt.set_index('TR')

# get spikes (trial x spike x unit)
# these are spike timestamps, and many entries in dimension 1 are NaN;
# it's filled only up to the number of spikes
spk = datmat['s'][0][0]
Nt, _, Nu = spk.shape

# make a list of categories for the stims:
categories = ['Faces', 'Animals', 'Bodies', 'Fruit', 'Natural', 'Manmade', 'Scene', 'Pattern']

# create a named tuple of observations
# movie is the stimulus and frame the time within that trial
# these conventions are to match the ethogram data and for use below
names = ['trial', 'frame', 'unit', 'count', 'movie', 'category']
Obs = namedtuple('Obs', names)

# find outliers
outliers = evt.index[evt.T_STIMOFF > 900].astype(int)
# outliers = np.r_[evt.index[evt.T_STIMOFF < 300], 
#                  evt.index[evt.T_STIMOFF > 400], 
#                  evt.index[evt.T_POST > 700]].astype(int)
outindex = np.unique(outliers)

# now loop over trials, then units within trial, counting spikes
# we count spikes in three intervals:
#   T_PRE:T_STIMON  pre-trial baseline
#   T_STIMON:T_STIMOFF  stimulus presentation
#   T_STIMOFF:T_POST  post-stimulus
tuplist = []  # accumulate observations here
for t in evt.itertuples():
    trial = int(t.Index)
    if (not fulldata) & (trial in outindex):
        continue
    if kk == 2:
        # kk = 2 is the shifted two-bin version
        epochs = [(t.T_PRE + 100, t.T_STIMON + 100), 
                    (t.T_STIMON + 100, t.T_STIMOFF + 100)]
    elif kk % 3 != 0:
        print('Error: Number of time bins: must be 2 or evenly divided by 3.')
        exit()
    else:
        landmk = np.array([t.T_PRE, t.T_STIMON, t.T_STIMOFF, t.T_POST])
        start = t.T_PRE
        epochs = []

        for index in range(landmk.shape[0] - 1):
            inter = (landmk[index + 1] - landmk[index]) * 3 / kk
            while start < landmk[index + 1]:
                epochs.append([start, start + inter])
                start += inter

    
    for unit in range(Nu):
        for epnum, ep in enumerate(epochs):
            t_spk = spk[trial - 1, :, unit]
            Nspk = np.sum((t_spk >= ep[0]) & (t_spk <= ep[1]))
            cat = categories[(int(t.STIM) - 1) // 12]
            if epnum == 0:  # baseline has no category
                cat = 'Baseline'
            tuplist.append(Obs(trial, epnum, unit, Nspk, int(t.STIM), cat))

# concatenate into dataframe
df = pd.DataFrame.from_records(tuplist, columns=names)
df = df.sort_values(by=['movie', 'frame', 'trial', 'unit'])

print('Getting movie start/stop times...')
Nstim = len(np.unique(df.movie))
N_uniq_times = kk * Nstim
# every stim is 3 epochs
part_frame = pd.DataFrame({'start': range(0, N_uniq_times, kk),
                           'end': range(kk - 1, N_uniq_times, kk)})
part_frame.to_csv('data/trial_start_end_times{}.csv'.format(kk), index=False)

# now transform (stimulus, time) pairs to unique times
print('Converting (stimulus, epoch) pairs to unique times...')
df_utimes = frames_to_times(df)

if fulldata:
    outfile = 'prepped_data{}full.csv'.format(kk)
else:
    outfile = 'prepped_data{}.csv'.format(kk)
    
outname = os.path.join(datadir, outfile)
df_utimes.to_csv(outname, index=False)
