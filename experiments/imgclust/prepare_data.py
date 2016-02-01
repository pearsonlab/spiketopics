from __future__ import division, print_function
import numpy as np
import scipy.io as sio
import pandas as pd
import os
from collections import namedtuple

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
names = ['trial', 'time', 'unit', 'count', 'stim']
Obs = namedtuple('Obs', names)

# now loop over trials, then units within trial, counting spikes
# we count spikes in three intervals:
#   -300:0  pre-trial baseline
#   0:300  within-trial baseline (300 < T_STIMON)
#   T_STIMON:T_STIMOFF  image display
tuplist = []  # accumulate observations here
for t in evt.itertuples():
    trial = int(t.Index)
    epochs = [(-300, 0), (0, 300), (t.T_STIMON, t.T_STIMOFF)]
    for unit in range(Nu):
        for epnum, ep in enumerate(epochs):
            t_spk = spk[trial - 1, :, unit]
            Nspk = np.sum((t_spk >= ep[0]) & (t_spk <= ep[1]))
            tuplist.append(Obs(trial, epnum, unit, Nspk, int(t.STIM)))

# concatenate into dataframe
df = pd.DataFrame.from_records(tuplist, columns=names)

outname = os.path.join(datadir, 'prepped_data.csv')
df.to_csv(outname, index=False)