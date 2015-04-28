"""
Prepare matrix of regressors for use with model fitting.
In this case, regressors are delta functions for each moment in the clip.
"""
from __future__ import division
import numpy as np
import pandas as pd
import scipy.sparse as sp
from helpers import frames_to_times

# load up data
datfile = './sql/spike_presentations.csv'

print "Reading data..."
df = pd.read_csv(datfile)

print "Processing data..."
df = df.drop(['trialId'], axis=1)
df = df.sort(['movieId', 'frameNumber', 'unitId'])
df = df.rename(columns={'unitId': 'unit', 'frameNumber': 'frame', 
    'movieId': 'movie', 'frameClipNumber': 'frame_in_clip'})
M = df.shape[0]

print "Converting (movie, frame) pairs to unique times..."
df = frames_to_times(df)

# set up regressors
# make binary regressor for frame in clip
print "Pivoting to form X..."
X = sp.coo_matrix((np.ones(M), (xrange(M), df['frame_in_clip'])))

print "Merging df and X..."
R = X.shape[1]
colnames = ['X' + str(r) for r in xrange(R)]
Xf = pd.DataFrame(X.toarray(), columns=colnames)

print "Dropping columns..."
allframe = pd.concat([df, Xf], axis=1)
allframe = allframe.drop('frame_in_clip', axis=1)

print "Saving data..."
outfile = 'data/spikes_plus_regressors.csv'
allframe.to_csv(outfile, index=False)