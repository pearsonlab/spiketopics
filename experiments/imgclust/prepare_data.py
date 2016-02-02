from __future__ import division, print_function
import numpy as np
import scipy.io as sio
import pandas as pd
import os
from collections import namedtuple
from spiketopics.helpers import frames_to_times, get_movie_partition

datadir = 'data'
fname = 'toroid20120718a.mat'
datfile = os.path.join(datadir, fname)

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

# create a named tuple of observations
# movie is the stimulus and frame the time within that trial
# these conventions are to match the ethogram data and for use below
names = ['trial', 'frame', 'unit', 'count', 'movie']
Obs = namedtuple('Obs', names)

# now loop over trials, then units within trial, counting spikes
# we count spikes in three intervals:
#   T_PRE:T_STIMON  pre-trial baseline
#   T_STIMON:T_STIMOFF  stimulus presentation
#   T_STIMOFF:T_POST  post-stimulus
tuplist = []  # accumulate observations here
for t in evt.itertuples():
    trial = int(t.Index)
    epochs = [(t.T_PRE, t.T_STIMON), (t.T_STIMON, t.T_STIMOFF),
    (t.T_STIMOFF, t.T_POST)]
    for unit in range(Nu):
        for epnum, ep in enumerate(epochs):
            t_spk = spk[trial - 1, :, unit]
            Nspk = np.sum((t_spk >= ep[0]) & (t_spk <= ep[1]))
            tuplist.append(Obs(trial, epnum, unit, Nspk, int(t.STIM)))

# concatenate into dataframe
df = pd.DataFrame.from_records(tuplist, columns=names)
df = df.sort_values(by=['movie', 'frame', 'trial', 'unit'])

print('Getting movie start/stop times...')
part_frame = get_movie_partition(df)
part_frame.to_csv('data/trial_start_end_times.csv', index=False)

# now transform (stimulus, time) pairs to unique times
print('Converting (stimulus, epoch) pairs to unique times...')
df_utimes = frames_to_times(df)

outname = os.path.join(datadir, 'prepped_data.csv')
df_utimes.to_csv(outname, index=False)
