import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def matshow(X, **kwargs):
    with sns.axes_style("white"):
        plt.matshow(X, aspect='auto', cmap='gray', **kwargs);

def frames_to_times(df):
    """
    Convert a dataframe with movie and frame columns to one with a single
    unique time index.
    """

    # make a dataframe of unique (movie, frame) pairs
    t_index = df[['movie', 'frame']].drop_duplicates().sort(['movie', 'frame'])
    t_index = t_index.reset_index(drop=True)  # get rid of original index
    t_index.index.name = 'time'  # name integer index
    t_index = t_index.reset_index()  # make time a column

    allframe = pd.merge(df, t_index).copy()  # merge time with original df
    allframe = allframe.drop(['movie', 'frame'], axis=1)  # drop cols

    return allframe

def gamma_from_hypers(mean_hyps, var_hyps, N=1e4):
    """
    Given a pair of hyperparameters for the mean and a pair for the variance parameter,
    draw N samples from the gamma distribution that results.
    """
    th = stats.gamma.rvs(a=mean_hyps[0], scale=1./mean_hyps[1], size=N)
    cc = stats.gamma.rvs(a=var_hyps[0], scale=1./var_hyps[1], size=N)
    
    aa = cc
    bb = cc * th
    
    return stats.gamma.rvs(a=aa, scale=1./bb)

def jitter_array(arr, percent):
    """
    Given an array, arr, return same array with each entry scaled by a 
    uniform percentage between +/- percent.
    """
    arr = np.array(arr)
    return arr * (1 + percent * (2 * np.random.rand(*arr.shape) - 1))

def jitter_inits(init_dict, percent_jitter):
    """
    Make dict of initial guesses for posterior parameters.
    percent_jitter is the variability of parameters around their values derived above
    """
    inits = {}
    keys = [k for k in init_dict if 'prior' in k]
    for key in keys:
        if key == 'xi':
            # initialize pretty much at random (10% 1's)
            rand_frac = 0.1
            xi_mat = (rand_frac >= np.random.rand(T, K))
            xi_mat = xi_mat.astype('float')
            xi_mat[:, 0] = 1  # assume a baseline (category 0) always present
            inits['xi'] = xi_mat
        else:
            inits[key] = jitter_array(init_dict[key], percent_jitter)
            
    # hack this for now
    # inits['bb'] = inits['aa'].copy()
    new_inits = init_dict.copy()
    new_inits.update(inits)
    return new_inits