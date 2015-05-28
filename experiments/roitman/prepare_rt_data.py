"""
Prepare data from Roitman and Shadlen 2002 for use with latent feature
model. Uses data from the reaction time version of the task, treating each
trial as unique (i.e., each "stimulus" observed by only one unit). This is
necessary because of varying durations of stimulus exposure/task pacing.
"""
from __future__ import division
import numpy as np
import pandas as pd
import scipy.io as scio

# load Roitman data from Matlab file
datfile = 'data/T1RT.mat'
dat = scio.loadmat(datfile, squeeze_me=True)

# column names for x variable: see ColumnNames608.m for explanation
x_col_names = ([
    'unit', 'stim_id', 'coherence', 'stim_dir', 'choice_dir', 'correct',
    't_stim_on', 't_saccade', 'rt', 'lat', 'novar', 'flash', 'seed', 
    't1_x', 't1_y', 't2_x', 't2_y', 'fix_x', 'fix_x', 'stim_x', 'stim_y',
    'vmax', 'vavg', 'off_x', 'off_y', 'beg_x', 'beg_y', 'end_x', 'end_y',
    't_stim_off', 'trial'
    ])

# X contains trial metadata: byteswap is needed for correct byte encoding
X = pd.DataFrame(dat['x'].byteswap().newbyteorder(), columns=x_col_names)

# S contains spikes for each trial
S = dat['s']

# shift times in S so that they are relative to stim onset
onsets = X['t_stim_on'].values
S -= onsets

# now bin spike times
dt = 20  # bin size in ms

# histogram each trial
chunklist = []
for idx in xrange(S.shape[0]):
    t_axis = np.arange(-100, X.loc[idx, 'rt'], 20)
    counts, _ = np.histogram(S[idx], bins=t_axis)
    chunk = pd.DataFrame({'count': counts, 'time': t_axis[:-1], 
        'unit': X.loc[idx, 'unit'], 'trial': X.loc[idx, 'trial']})
    chunklist.append(chunk)

# make a "long" dataframe of observations
countframe = pd.concat(chunklist)

# consider all trials as unique stimuli; create unique time index
countframe = countframe.reset_index(drop=True)  # make unique index each obs
countframe = countframe.reset_index()  # make it a column
countframe = countframe.rename(columns={'index': 'utime'})  # rename it 'utime'

# write out to disk
outfile = 'data/roitman_rt_data.csv'
countframe.to_csv(outfile, index=False)
