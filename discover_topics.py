"""
Fit Gamma-Poisson topic model to movie/ethogram dataset.
"""
from __future__ import division
import numpy as np
import pandas as pd
import gamma_poisson as gp

# load up data
datfile = './sql/spike_presentations.csv'
df = pd.read_csv(datfile)
df = df.drop(['trialId', 'frameClipNumber'], axis=1)
df = df.sort(['movieId', 'frameNumber', 'unitId'])
df = df.rename(columns={'unitId': 'unit', 'frameNumber': 'frame', 
    'movieId': 'movie'})

# set up params 
dt = 1 / 30  # duration of movie frame
T = df[['movie', 'frame']].drop_duplicates().shape[0]
U = df['unit'].drop_duplicates().shape[0]
K = 5

# set up model object
gpm = gp.GPModel(T, K, U, dt)
gpm = gpm.set_data(df)