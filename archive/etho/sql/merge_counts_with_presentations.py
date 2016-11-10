"""
Merge counts data with presentations data. Write out as cleaned dataframe.
"""
import pandas as pd

pres = pd.read_csv('presentations.csv')
spks = pd.read_csv('spikecounts.csv')

df = pd.merge(spks, pres, how='right')

df = df.sort(['unitId', 'trialId', 'frameClipNumber'])
df = df.fillna(0)  # fill missing counts with 0
df = df.reset_index(drop=True)  # get rid of meaningless index

outfile = 'spike_presentations.csv'
df.to_csv(outfile, index=False)