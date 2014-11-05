from __future__ import division
import sqlalchemy
import pandas as pd
import numpy as np
import pandas.io.sql as sql
import subprocess
import sys

def init_sql(connectstr):
    # check if mysql is running
    outstr = subprocess.check_output(['ps', '-A'])

    if outstr.find('sql') == -1:
        print "MySQL not running!"
        eng = None
    else:
        eng = sqlalchemy.create_engine('mysql://root@localhost/surfer')

    return eng

def get_all_runs_all_units(eng):
    query_string = """
    SELECT DISTINCT unitId, plexonRunId 
    FROM UnitRuns;
    """
    return sql.read_sql_query(query_string, eng)

def get_events(unit, run, eng):
    event_query = """
    SELECT unitId, trialId, plxMovieDrawOn
    FROM UnitTrials
    WHERE unitId = {} AND plexonRunId = {};
    """.format(unit, run)

    return sql.read_sql_query(event_query, eng)

def get_spikes(unit, run, eng):
    spike_query = """
    SELECT spikeTime, unitId FROM CleanSpikes
    WHERE unitId = {} AND plexonRunId = {};
    """.format(unit, run)

    return sql.read_sql_query(spike_query, eng)

def match_spikes_to_trials(trial_start_times, spikes):
    """
    Given a dataframe of events (from get_events) and a DataFrame of
    spikes from get_spikes, return a dataframe like spikes with an 
    additional column containing the start time of the trial into which
    the given spike falls
    """
    # figure out which trial each spike belongs to
    which_trial = np.digitize(spikes['spikeTime'], trial_start_times)

    # expand start times to include a row for times before trial 1
    start_times_aug = np.concatenate([[0], trial_start_times])

    # make a dataframe of start times
    start_times_df = pd.DataFrame(start_times_aug[which_trial], 
        columns=['plxMovieDrawOn'])
    return pd.concat([spikes, start_times_df], axis=1)

def merge_spikes_with_events(spks, events):
    """
    Given a dataframe with spikes assigned to start times, 
    fill in event information for each spike.
    """
    # merge on clip start time
    return pd.merge(events, spks, how='right', sort=True)

def calculate_relative_spike_times(df, fps):
    """
    Calculate the time of each spike relative to the start of trial.
    Returns time in frames relative to trial start.
    df contains one spike per row, with a timestamp and a trial start time.
    fps is the number of frames per second. 
    """
    dat = df.copy()
    # make new columns for time and frame within trial
    dat['time_in_trial'] = dat['spikeTime'] - dat['plxMovieDrawOn']
    dat['frame_in_trial'] = (dat['time_in_trial'] * fps).astype('int') 
    return dat

def count_spikes(df):
    """
    df has one row per spike. Group by trial and frame in trial and 
    return spike count in a dataframe.
    """
    # count spikes per frame
    grp = df.groupby(['unitId', 'trialId', 'frame_in_trial'])
    spkcount = grp.count()['time_in_trial'].reset_index()
    spkcount = spkcount.rename(columns={'time_in_trial': 'count'})
    return spkcount

def get_spike_counts(unit, run, eng):
    """
    For a given unit and run, extract all spike counts in a 
    DataFrame with unit, trial, frame, and count as columns
    """
    spikes = get_spikes(unit, run, eng)
    events = get_events(unit, run, eng)

    spkdf = match_spikes_to_trials(events['plxMovieDrawOn'], spikes)

    spkmerge = merge_spikes_with_events(spkdf, events)

    spkmerge = calculate_relative_spike_times(spkmerge, frames_per_second)

    spkcount = count_spikes(spkmerge)

    return spkcount

if __name__ == '__main__':

    connectstr = 'mysql://root@localhost/surfer'
    frames_per_second = 30

    # set up database connection
    eng = init_sql(connectstr)
    if not eng:
        sys.exit('Closing shop')
    
    # get all combinations of units and runs        
    unit_run_df = get_all_runs_all_units(eng)

    # initialize dataframe
    df = pd.DataFrame()

    for _, row in unit_run_df.iterrows():
        unit = row['unitId']
        run = row['plexonRunId']
        print "unit {}, run {}".format(unit, run)
        spkcount = get_spike_counts(unit, run, eng)
        df = df.append(spkcount)

    # clean up and write out
    outfile = 'spikecounts.csv'
    df[['unitId', 'trialId']] = df[['unitId', 'trialId']].astype('int')
    df.to_csv(outfile)