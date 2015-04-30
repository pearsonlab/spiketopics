"""
Prepare matrix of regressors for use with model fitting.
In this case, regressors are delta functions for each moment in the clip.
"""
from __future__ import division
import numpy as np
import pandas as pd
import scipy.sparse as sp
from helpers import frames_to_times, regularize_zeros
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare regressors for running VB model')
    parser.add_argument('collection', help='collection of regressors to use', nargs='?', choices=['mean', 'full'], default='mean')

    args = parser.parse_args()
    print 'Preparing {} collection of regressors'.format(args.collection)

    # load up data
    datfile = './sql/spike_presentations.csv'

    print 'Reading data...'
    df = pd.read_csv(datfile)

    print 'Processing data...'
    df = df.drop(['trialId'], axis=1)
    df = df.sort(['movieId', 'frameNumber', 'unitId'])
    df = df.rename(columns={'unitId': 'unit', 'frameNumber': 'frame', 
        'movieId': 'movie', 'frameClipNumber': 'frame_in_clip'})
    M = df.shape[0]

    # and renumber units consecutively (starting at 0)
    df['unit'] = np.unique(df['unit'], return_inverse=True)[1]

    print 'Converting (movie, frame) pairs to unique times...'
    df = frames_to_times(df)

    if args.collection == 'mean':
        umean = df.groupby(['unit', 'frame_in_clip']).mean().reset_index()
        Xmean = pd.pivot_table(umean, index='frame_in_clip', 
            columns='unit', values='count')

        # put Xmean on log scale 
        Xm = regularize_zeros(Xmean)
        X0 = np.log(Xm)
        X0 -= X0.min()

        # # now take derivatives
        # # X1 = d(log Xmean)/dt
        # X1 = Xmean.diff() / Xm
        # X1 -= X1.min()
        # X1 = regularize_zeros(X1)
        # X2 = regularize_zeros(X1.diff())
        # X2 -= X2.min()

        # append to data frame
        uu = df['unit']
        fic = df['frame_in_clip'] 
        df['X0'] = X0.values[fic, uu]
        # df['X1'] = X1.values[fic, uu]
        # df['X2'] = X2.values[fic, uu]

        R = 1
        allframe = df
        outfile = 'data/spikes_plus_minimal_regressors.csv'

    elif args.collection == 'full':
        # set up regressors
        # make binary regressor for frame in clip
        print 'Pivoting to form X...'
        X = sp.coo_matrix((np.ones(M), (xrange(M), df['frame_in_clip'])))

        outfile = 'data/spikes_plus_regressors.csv'

        print 'Merging df and X...'
        R = X.shape[1]
        colnames = ['X' + str(r) for r in xrange(R)]
        Xf = pd.DataFrame(X.toarray(), columns=colnames)
        allframe = pd.concat([df, Xf], axis=1)

    print 'Dropping columns...'
    allframe = allframe.drop('frame_in_clip', axis=1)

    print 'Saving data...'
    allframe.to_csv(outfile, index=False)