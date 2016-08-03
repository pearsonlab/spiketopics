import scipy.io as sio
import os
import pandas as pd
#from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
#from spiketopics.helpers import frames_to_times, get_movie_partition

datadir = 'data'
fname = 'toroid20120718a.mat'
datfile = os.path.join(datadir, fname)

# load data
datmat = sio.loadmat(datfile)['dat']

# get header
header = datmat['h'][0][0]
evtdat = datmat['c'][0][0]
evtnames = header['vv'][0][0][0].dtype.names
evt = pd.DataFrame(evtdat, columns=evtnames[:10])
evt = evt.set_index('TR')

spk = datmat['s'][0][0]
Nt, _, Nu = spk.shape
header = datmat['h'][0]

# input the unit names for plot titles (or can be extracted from header)
snames = ['sig057a','sig065a','sig065b','sig066a','sig068a','sig068b','sig069a','sig070a','sig070b','sig072a',
'sig073a','sig074a','sig074b','sig078a','sig080a','sig081a','sig082a','sig082b','sig083a','sig085a','sig086a',
'sig086b','sig087a','sig088a','sig089a','sig091a','sig092a','sig094a','sig095a','sig096a','sig097a','sig100a',
'sig101a','sig101b','sig102a','sig105a','sig108a','sig112a','sig113a','sig113b','sig114a','sig115a','sig116a',
'sig117a','sig117b','sig118a','sig119a','sig121a','sig122a','sig122b','sig125a','sig126a','sig126b','sig127a',
'sig128a','sig128b']

# To generate raster plots across all units in trials
# First loop over units, then trials to create tuplistunit
# Finally determine which subplot shall we plot on

# print to pdf using PdfPages
pp = PdfPages('rasterplot_unit.pdf')

for unit in range(Nu):
    # plot figures with proper names
    plt.figure(figsize = (32,20))
    plt.suptitle(snames[unit], fontsize = 48)
    # generate empty tuple list
    tuplistunit = []
    # sum across trials
    for trial in range(Nt):
        t_spk = spk[trial, :, unit]
        tuplistunit.append(t_spk[np.where(t_spk > -20525)])
    
    # record the number of trials within each image
    plotcnt = np.zeros(96)
    for ith, trial in enumerate(tuplistunit):
        # 12x8 subplots
        plt.subplot(8, 12, evt.STIM[ith+1])
        plotidx = evt.STIM[ith+1] - 1
        plotcnt[plotidx] += 1 
        # landmarks of stimulus
        landmarks = [evt.T_PRE[ith+1], evt.T_STIMON[ith+1], evt.T_STIMOFF[ith+1], evt.T_POST[ith+1]]
        plt.vlines(landmarks, plotcnt[plotidx] - 0.5, plotcnt[plotidx] + 0.5, color='r', linewidth=1)
        # plot actual spikes
        plt.vlines(trial, plotcnt[plotidx] - 0.5, plotcnt[plotidx] + 0.5, color='k')
        plt.ylim(.5, plotcnt[plotidx] + .5)
        #plt.xlabel('Time')
        #plt.ylabel('Trial')
    
    pp.savefig()
    
pp.close()

# There are 8 categories and each contains 12 imagies. 
# The x-axis are time (ms) and y-axis are trial numbers.
# Black lines are the place where a spike triggers, and 
#   red lines are the landmarked times for each trial, i.e. T_PRE, T_STIMON, T_STIMOFF and T_POST
# There are three trials across each unit have extremely long T_STIMOFF and T_POST times.