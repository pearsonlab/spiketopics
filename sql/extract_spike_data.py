import sqlalchemy
import pandas as pd
import numpy as np
import pandas.io.sql as sql
import subprocess
import sys

unit = 44
plxrun = 33

event_query = """
SELECT unitId, trialId, plxMovieDrawOn
FROM UnitTrials
WHERE unitId = {} AND plexonRunId = {};
""".format(unit, plxrun)

spike_query = """
SELECT spikeTime, unitId FROM CleanSpikes
WHERE unitId = {} AND plexonRunId = {};
""".format(unit, plxrun)

# check if mysql is running
outstr = subprocess.check_output(['ps', '-A'])

if outstr.find('sql') == -1:
    sys.exit("MySQL not running!")
else:
    eng = sqlalchemy.create_engine('mysql://root@localhost/surfer')

events = sql.read_sql_query(event_query, eng)
spikes = sql.read_sql_query(spike_query, eng)

# figure out which trial each spike belongs to
trial_edges = events['plxMovieDrawOn']
which_trial = np.digitize(spikes['spikeTime'], trial_edges)

# merge spike data frame with events dataframe on start time
trial_start_times = pd.DataFrame(np.concatenate([[0], trial_edges])[which_trial], columns=['plxMovieDrawOn'])
spikes_plus_times = pd.concat([spikes, trial_start_times], axis=1)
evts_plus_spikes = pd.merge(events, spikes_plus_times, how='outer', sort=True)

# make new columns for time and frame within trial
evts_plus_spikes['time_in_trial'] = evts_plus_spikes['spikeTime'] - evts_plus_spikes['plxMovieDrawOn']
frames_per_second = 30
evts_plus_spikes['frame_in_trial'] = (evts_plus_spikes['time_in_trial'] * frames_per_second).astype('int') 

# count spikes per frame
spkcount = evts_plus_spikes.groupby(['trialId', 'frame_in_trial']).count()['time_in_trial'].reset_index()
spkcount = spkcount.rename(columns={'time_in_trial': 'count'})