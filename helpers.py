import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import copy


def matshow(X, **kwargs):
    with sns.axes_style("white"):
        ax = plt.matshow(X, aspect='auto', cmap='gray', **kwargs);
    return ax

def rle(x):
    """
    Perform run length encoding on the numpy array x. Returns a tuple
    of start indices, run lengths, and values for each run.
    """
    # add infinity to beginning to x[0] always starts a run
    dx = np.diff(np.insert(x.astype('float').flatten(), 0, np.inf))
    starts = np.flatnonzero(dx)
    # make sure stops always include last element
    stops = np.append(starts[1:], np.size(x))
    runlens = stops - starts
    values = x[starts]
    return starts, runlens, values

def mutual_information_matrix(X, Z):
    """
    Given X, an observations x variables array of observed binary variables and
    Z, variables x probabilities, p(z = 1), construct a matrix of 
    (normalized) mutual information scores. Output shape is 
    X.shape[1] x Z.shape[1]
    """
    Nx = X.shape[1]
    Nz = Z.shape[1]
    mi_mat = np.empty((Nx, Nz))
    for i in xrange(Nx):
        for j in xrange(Nz):
            mi_mat[i, j] = mi(X[:, i], Z[:, j])

    return mi_mat

def mi(x, z):
    """
    Calculate mutual information (normalized between 0 and 1) between 
    an observed binary sequence x and a sequence of posterior probabilities
    z. 
    """
    zm = np.mean(z)
    xm = np.mean(x)
    Hz = -zm * np.log(zm) - (1 - zm) * np.log(1 - zm)
    Hx = -xm * np.log(xm) - (1 - xm) * np.log(1 - xm)
    x0 = x == 0
    x1 = x == 1
    p1 = np.mean(x)
    p0 = 1 - p1
    pz0 = np.mean(z[x0]) 
    pz1 = np.mean(z[x1])
    Hzx = -p0 * (pz0 * np.log(pz0) + (1 - pz0) * np.log(1 - pz0))
    Hzx += -p1 * (pz1 * np.log(pz1) + (1 - pz1) * np.log(1 - pz1))
    
    return (Hz - Hzx) / np.sqrt(Hz * Hx)

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

def regularize_zeros(df):
    """
    Given a DataFrame with 0s and NAs, regularize it so that it can be 
    put on a log scale.
    """
    new_df = df.copy()
    new_df[new_df == 0.0] = np.nan
    new_mins = new_df.min()
    new_df = new_df.fillna(new_mins)

    return new_df

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

def lognormal_from_hypers(mu, scale, shape, rate, N=1e4):
    """
    Draw N samples from a lognormal distribution whose (m, s) parameters
    are drawn from a normal-gamma hyperprior with parameters 
    (mu, scale, shape, rate)
    """
    tau = stats.gamma.rvs(a=shape, scale=1./rate, size=N)
    sig = 1. / np.sqrt(tau)
    s = 1. / np.sqrt(scale * tau)
    m = stats.norm.rvs(loc=mu, scale=s, size=N)
    x = stats.lognorm.rvs(scale=np.exp(m), s=sig, size=N)
    
    return x

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
    keys = [k for k in init_dict if 'post' in k or 'init' in k]
    for key in keys:
        if key == 'z_init':
            old_z = init_dict['z_init']
            xi_mat = np.random.rand(*old_z.shape)
            xi_mat = xi_mat.astype('float')
            z_init = xi_mat / np.sum(xi_mat, axis=0, keepdims=True)
            inits['z_init'] = z_init
        elif key == 'zz_init':
            inits['zz_init'] = np.random.rand(*init_dict['zz_init'].shape)
        else:
            inits[key] = jitter_array(init_dict[key], percent_jitter)
            
    # hack this for now
    new_inits = copy.deepcopy(init_dict)
    new_inits.update(inits)
    return new_inits