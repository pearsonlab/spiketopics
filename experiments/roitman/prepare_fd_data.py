"""
Prepare data from Roitman and Shadlen 2002 for use with latent feature
model. Uses data from the fixed duration version of the task. Treats
every stimulus as distinct, but otherwise trials are assumed identical.
"""
from __future__ import division
import numpy as np
import pandas as pd
import scipy.io as scio
import os
import re

def process_file(fname, unit):
    """
    Process data file given by fname. Assign the data inside to unit, 
    where unit is a unit number. Return a dataframe of spike counts and 
    trial variables in "long" form.
    """
    # load matlab file
    dat = scio.loadmat(fname, squeeze_me=True)['data']

    # pick which columns we want to keep
    cols_to_grab = ([
        'trial', 'stimulus', 'coherence', 'fixation_on', 'fixation', 'target_on', 
        'stim_on', 'stim_off', 'fixation_off', 'saccade',  'reward',
        'choice', 'correct'
        ])

    # indices of columns (-1 since Python starts at 0)
    idx_to_grab = np.array([2, 3, 5, 39, 31, 32, 34, 35, 36, 37, 40, 12, 13]) - 1

    # make it a data frame: all times are in ms relative to fixation on
    X = pd.DataFrame(dat[1][:, idx_to_grab].byteswap().newbyteorder(), columns=cols_to_grab)

    # filtering operations
    ## take only stim periods near 1s
    stim_dur = X['stim_off'] - X['stim_on']
    X = X[stim_dur > 900]
    ## take only delay periods longer than 500
    delay = X['fixation_off'] - X['stim_off']
    # X = X[delay > 700]

    dt = 20  # bin size in ms
    # countframe = make_count_frame(dat, X, dt, 'stim_on', 'stim_off', -700, 500)
    countframe = make_count_frame(dat, X, dt, 'stim_on', 'stim_off', -0, 0)
    if countframe is not None:
        countframe['unit'] = unit

    return countframe

def make_count_frame(dat, event_frame, dt, start_evt, stop_evt, start, stop):
    """
    Construct a dataframe of counts and events. 
    dat is the matlab data structure
    event_frame is the event dataframe
    dt is the time bin size (in ms)
    start_evt is a string specifying the event that is time 0
    stop_evt is a string specifying the event relative to which time is stopped
    start is the time relative to fixation point on (negative = before)
    stop is the time relative to saccade
    Returns a dataframe like event frame, but with a relative time axis and 
        spike counts
    """
    # now loop over rows (ugh), reconciling spike times and histogramming
    chunklist = []
    for idx, row in event_frame.iterrows():

        # test if stimulus was displayed (non-NaN coherence)
        # and if there were no recording problems (errstatus = False)
        errstatus = len(dat[2][idx, 3]) != 0
        if not np.isnan(row['coherence']) and not errstatus:
            # get spike times
            spikes = dat[2][idx, 0]

            # time of spike clock start (relative to fixation point on)
            spike_clock_start = dat[2][idx, 1]  

            # reconcile to fixation point onset
            spikes -= spike_clock_start + row[start_evt]

            # taxis start ms before fixation point to stop ms after stim_off
            dur = row[stop_evt] - row[start_evt]
            t_axis = np.arange(start, dur + stop, dt)

            # bin, make data frame, append to chunklist
            counts, _ = np.histogram(spikes, bins=t_axis)
            chunk = pd.DataFrame({'count': counts, 'time': t_axis[:-1], 
                'trial': row['trial']})
            chunklist.append(chunk)

    # make a "long" dataframe of observations
    try:
        countframe = pd.concat(chunklist)
        countframe = countframe.merge(event_frame, on='trial')
        return countframe
    except:
        return None

def is_during_stim(df, origin):
    """
    Make binary/Boolean indicator series for whether each row is
    during stim presentation.
    origin is a string specifying the event that is time 0
    """
    origin = df[origin].iloc[0]
    stim_on = df['stim_on'].iloc[0] - origin
    stim_off = df['stim_off'].iloc[0] - origin
    X = (df['time'] > stim_on) & (df['time'] <= stim_off)
    X.index = df['time'].copy() 
    X.name = 'Xstim'
    return X

def is_peri_stim(df, origin, interval):
    """
    Make binary/Boolean indicator series for whether each row is
    is in a window around stim presentation. Negative numbers are
    before onset, postive numbers after offset.
    origin is a string specifying the event that is time 0
    """
    origin = df[origin].iloc[0]
    stim_on = df['stim_on'].iloc[0] - origin
    stim_off = df['stim_off'].iloc[0] - origin
    if interval < 0:
        X = (df['time'] > stim_on + interval) & (df['time'] <= stim_on)
    elif interval > 0: 
        X = (df['time'] > stim_off) & (df['time'] <= stim_off + interval)
    X.index = df['time'].copy() 
    X.name = 'Xperi' + str(int(interval))
    return X

def normalize_times(df):
    """
    Convert (stimulus, relative time) pairs in dataframe to a unique time 
    index.
    """

    t_index = df[['stimulus', 'time']].drop_duplicates().sort(['stimulus', 'time'])
    t_index = t_index.reset_index(drop=True)  # get rid of original index
    t_index.index.name = 'utime'  # name integer index
    t_index = t_index.reset_index()  # make time a column

    allframe = pd.merge(df, t_index).copy()  # merge time with original df
    # allframe = allframe.drop(['stimulus', 'time'], axis=1)  # drop cols

    return allframe

if __name__ == '__main__':
    
    # the following regex matches the pattern 
    # [letters][unit number][letters].mat
    file_regex = '^[a-zA-Z]+(?P<unit>\d+)[a-zA-Z]*.mat$'
    pat = re.compile(file_regex)

    # load Roitman data from Matlab files
    datadirs = ['b_fd', 'n_fd']
    df_list = []

    for d in datadirs:
        # walk directories
        for dirpath, _, fnames in os.walk(os.path.join('data', d)):
            # iterate over files
            for fn in fnames:
                # check if filename matches pattern
                result = pat.search(fn)
                if result:
                    # extract unit from filename
                    unit = result.groupdict()['unit']
                    fn_full = os.path.join(dirpath, fn)

                    # make count frame and append to df_list
                    df_list.append(process_file(fn_full, unit))

    countframe = pd.concat(df_list)

    # make boxcar regressor for stim on
    censored_stim = lambda x: is_during_stim(x, 'stim_on')
    Xstim = countframe.groupby(['unit', 'trial']).apply(censored_stim)
    Xstim = Xstim.reset_index()

    # # make boxcar for period before stim on
    # grab_fun = lambda x: is_peri_stim(x, 'stim_on', -700)
    # Xpre = countframe.groupby(['unit', 'trial']).apply(grab_fun)
    # Xpre.name = 'Xpre'
    # Xpre = Xpre.reset_index()

    # # make boxcar for period after stim on
    # grab_fun = lambda x: is_peri_stim(x, 'stim_on', 500)
    # Xpost = countframe.groupby(['unit', 'trial']).apply(grab_fun)
    # Xpost.name = 'Xpost'
    # Xpost = Xpost.reset_index()

    # Xframe = Xstim.merge(Xpre).merge(Xpost)
    # Xframe = Xstim.merge(Xpre)
    Xframe = Xstim

    countframe = countframe.merge(Xframe)

    # for some reason, the above merge introduces duplicates
    countframe = countframe.drop_duplicates() 

    # make regressors for each coherence level
    clevels = countframe['coherence'].unique()
    clevels.sort()
    for coh in clevels:
        countframe['Xcoh' + str(int(coh))] = (
            (countframe['coherence'] == coh) & countframe['Xstim'])

    # make regressors for stimulus direction (into (True) vs out of RF (False))
    is_stim_and_correct = countframe['Xstim'] & countframe['correct']
    correct = countframe['correct'].astype('bool')
    chose_out_RF = (countframe['choice'] - 1).astype('bool')
    chose_into_RF = ~chose_out_RF
    countframe['Xinto'] = is_stim_and_correct & chose_into_RF
    countframe['Xout'] = is_stim_and_correct & chose_out_RF
    countframe['into_RF'] = ((chose_into_RF & correct) |
        chose_out_RF & ~correct).astype('int')

    # convert to unique times
    countframe = normalize_times(countframe)

    # write out to disk
    outfile = 'data/roitman_fd_data.csv'
    countframe.to_csv(outfile, index=False)
