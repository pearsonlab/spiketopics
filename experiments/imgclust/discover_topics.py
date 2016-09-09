"""
Fit Gamma-Poisson topic model to McMahon-Leopold dataset.
"""
from __future__ import division
import numpy as np
import pandas as pd
import spiketopics.gamma_model as gp
from spiketopics.helpers import jitter_inits
import argparse
import cPickle as pickle

if __name__ == '__main__':

    # Modified by Xin
    parser = argparse.ArgumentParser(description="Run latent topic discovery using a Gamma-Poisson model")
    parser.add_argument("input", help="name of input file")
    parser.add_argument("-s", "--seed", help="random number seed",
        default=12345)
    
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed)
    
    # load up data
    datfile = args.input
    print "Reading data..."
    df = pd.read_csv(datfile)

    ###########################################
    # For testing only
    # No Need to input datafile in the argument
    # np.random.seed(101)
    # print "Reading data..."
    # df = pd.read_csv("data/prepped_data.csv")

    ####### for testing only ################
    # print "Subsetting for testing..."
    # df = df.query('utime <= 1e5')
    ####### for testing only ################

    # set up params
    print "Calculating parameters..."

    # Modified by Xin: since we also need the trial info 
    # to compute natural time correlations
    # juggle columns
    # df = df[['time', 'unit', 'count']]

    # and renumber units consecutively (starting at 0)
    df['unit'] = np.unique(df['unit'], return_inverse=True)[1]

    M = df.shape[0]
    T = df['time'].drop_duplicates().shape[0]
    U = df['unit'].drop_duplicates().shape[0]
    R = df.shape[1] - len(['time', 'unit', 'count'])
    K = 10
    D = 100  # maximum semi-Markov duration
    Mz = 2  # number of levels of each latent state
    time_natural = int(T / (12 * 8))  # the number of natural time bins within each trial
    print "There are {} time bins per trial".format(time_natural)
    # dt = df.loc[1, 'time'] - df.loc[0, 'time']  # duration of bin
    dt = 900 / time_natural  # 300 or 150 ms bins
    od_natural = True  # flag for overdispersion_natural

    #################### priors and initial values

    print "Setting priors and inits..."

    ############ baseline ####################
    bl_mean_shape = 2.
    bl_mean_rate = 40 * dt  # actual parameter should be per-bin rate
    bl_shape_shape = 30.
    bl_shape_rate = 30.


    baseline_dict = ({
                'prior_shape_shape': bl_shape_shape,
                'prior_shape_rate': bl_shape_rate,
                'prior_mean_shape': bl_mean_shape,
                'prior_mean_rate': bl_mean_rate,
                'post_shape_shape': bl_shape_shape,
                'post_shape_rate': bl_shape_rate,
                'post_mean_shape': bl_mean_shape,
                'post_mean_rate': bl_mean_rate,
                'post_child_shape': np.ones((U,)),
                'post_child_rate': np.ones((U,))
                })

    ############ firing rate latents ####################
    fr_shape_shape = 2. * np.ones((K,))
    fr_shape_rate = 1e-4 * np.ones((K,))
    # fr_shape_shape = 1. * np.ones((K,))
    # fr_shape_rate = 1 * np.ones((K,))
    fr_mean_shape = 1 * U * np.ones((K,))
    fr_mean_rate = 1 * U * np.ones((K,))
    # fr_mean_shape = 1 * np.ones((K,))
    # fr_mean_rate = 1 * np.ones((K,))

    sp_prior = np.array([1, 1])
    sp_init = np.array([10, 1])
    fr_latent_dict = ({
                'prior_shape_shape': fr_shape_shape,
                'prior_shape_rate': fr_shape_rate,
                'prior_mean_shape': fr_mean_shape,
                'prior_mean_rate': fr_mean_rate,
                'post_shape_shape': fr_shape_shape,
                'post_shape_rate': fr_shape_rate,
                'post_mean_shape': fr_mean_shape,
                'post_mean_rate': fr_mean_rate,
                'post_child_shape': np.ones((U, K)),
                'post_child_rate': np.ones((U, K))})
                # 'prior_sparsity': np.tile(sp_prior.reshape(2, 1), (1, K)),
                # 'post_sparsity': np.tile(sp_init.reshape(2, 1), (1, K)),
                # 'post_features': 0.8 * np.random.rand(U, K),
                # 'spike_severity': 1e8
                # })

    ############ latent states ####################
    ###### A ###############
    # A_off = 10.
    # A_on = 1.
    # Avec = np.r_[A_off, A_on].reshape(2, 1, 1)
    # A_prior = np.tile(Avec, (1, 2, K))
    A_cat = (10 * np.eye(2) + 1).reshape(2, 2, 1)
    A_prior = np.tile(A_cat, (1, 1, K))

    ###### pi ###############
    pi_off = 150.
    pi_on = 1.
    pi_prior = np.tile(np.r_[pi_off, pi_on].reshape(2, 1), (1, K))

    ###### p(d) #############
    d_hypers = (2.5, 4., 2., 40.)
    d_pars = ({'d_prior_mean': d_hypers[0] * np.ones((Mz, K)),
              'd_prior_scaling': d_hypers[1] * np.ones((Mz, K)),
              'd_prior_shape': d_hypers[2] * np.ones((Mz, K)),
              'd_prior_rate': d_hypers[3] * np.ones((Mz, K))})
    d_inits = (3., 1., 1., 1.)

    d_post_pars = ({'d_post_mean': d_inits[0] * np.ones((Mz, K)),
                    'd_post_scaling': d_inits[1] * np.ones((Mz, K)),
                    'd_post_shape': d_inits[2] * np.ones((Mz, K)),
                    'd_post_rate': d_inits[3] * np.ones((Mz, K))})


    # E[z]
    # initialize pretty much at random (10% 1's)
    rand_frac = 0.01
    xi_mat = (rand_frac >= np.random.rand(T, K))
    xi_mat = xi_mat.astype('float')
    z_prior = np.dstack([1 - xi_mat, xi_mat]).transpose((2, 0, 1))

    # E[zz]
    Xi_mat = np.random.rand(2, 2, T - 1, K)
    Xi_mat = Xi_mat.astype('float')

    # for parallel processing, we need a list of tuples defining start and
    # end times of chunks
    movie_df = pd.read_csv('data/trial_start_end_times{}.csv'.
        format(str(time_natural)))
    chunklist = zip(movie_df['start'], movie_df['end'])

    # now make sure entries in Xi corresponding to p(z_start, z_previous_stim) don't count in Xi
    start_list = [s for s, _ in chunklist if s > 0]
    for t in start_list:
        Xi_mat[:, :, t - 1] = 0

    latent_dict = ({'A_prior': A_prior, 'pi_prior': pi_prior,
                    'A_post': A_prior, 'pi_post': pi_prior,
                    'z_init': z_prior, 'zz_init': Xi_mat,
                    'logZ_init': np.zeros((K,)),
                    'chunklist': chunklist
                    })
    # latent_dict.update(d_pars)
    # latent_dict.update(d_post_pars)
    ############ regression coefficients ####################
    # ups_shape = 11.
    # ups_rate = 10.
    # reg_shape = ups_shape * np.ones((U, R))  # shape
    # reg_rate = ups_rate * np.ones((U, R))  # rate

    # # since we know exact update for a_mat, use that
    # nn = df['count']
    # uu = df['unit']
    # NX = nn[:, np.newaxis] * df.iloc[:, -R:]
    # NX *= 0.5  # don't start off too big
    # a_mat = NX.groupby(uu).sum().values + reg_rate
    # b_mat = a_mat.copy()

    # reg_dict = ({'prior_shape': reg_shape, 'prior_rate': reg_rate,
    #             'post_shape': a_mat, 'post_rate': b_mat
    #              })

    ############ overdispersion ####################
    od_shape = 6.
    od_rate = 6.
    od_dict = ({'prior_shape': od_shape * np.ones((M,)),
                'prior_rate': od_rate * np.ones((M,)),
                'post_shape': np.ones((M,)),
                'post_rate': np.ones((M,))
                })


    ############ overdispersion natural time: Xin ####################
    od_natural_shape = 6.
    od_natural_rate = 6.
    od_natural_dict = ({
                'prior_shape': od_natural_shape * (np.ones(
                    (M,)).reshape(-1, time_natural) * np.array([1, 10, 20])).ravel(),
                'prior_rate': od_natural_rate * (np.ones(
                    (M,)).reshape(-1, time_natural) * np.array([1, 10, 20])).ravel(),
                'post_shape': np.ones((M,)),
                'post_rate': np.ones((M,))
                })


    ############ initialize model ####################
    numstarts = 5
    fitobjs = []
    Lvals = []
    for idx in xrange(numstarts):
        # set up model object
        gpm = gp.GammaModel(df, K)
        gpm.initialize_baseline(**jitter_inits(baseline_dict, 0.25))
        gpm.initialize_fr_latents(**jitter_inits(fr_latent_dict, 0.15))
        gpm.initialize_latents(**jitter_inits(latent_dict, 0.25))
        # gpm.initialize_fr_regressors(**jitter_inits(reg_dict, 0.25))

        # Modified by Xin
        if not od_natural:
            gpm.initialize_overdispersion(**jitter_inits(od_dict, 0.25))
        else:
            gpm.initialize_overdispersion_natural(**jitter_inits(od_natural_dict, 0.25))
            gpm.time_natural = time_natural

        gpm.finalize()

        print "Start {} -----------------------".format(idx)
        gpm.do_inference(tol=1e-5, verbosity=3)
        print "Final L = {}".format(gpm.L())
        Lvals.append(gpm.L())
        fitobjs.append(gpm)

    ############## fit model
    print "Choosing best model..."
    # pick out best fit
    bestind = np.argmax(Lvals)
    gpm = fitobjs[bestind]
    del fitobjs  # to save memory

    print "Cleaning up..."
    # need to get rid of externally defined functions for pickling
    # also has the effect of "neutering" object from futher update
    for name, node in gpm.nodes.iteritems():
        attrs = node.__dict__.keys()
        to_delete = [a for a in attrs if 'update' in a or 'log_prior' in a]
        for a in to_delete:
            delattr(node, a)

    # for semi-Markov, more stuff to delete
    try:
        dnode = gpm.nodes['HMM'].nodes['d']
        to_delete = ['logpd', 'calc_ess', 'update']
        for fn in to_delete:
            delattr(dnode, fn)
    except:
        pass

    print "Writing to disk..."
    # fstub = '_{}K_{}S'.format(K, numstarts)
    fstub = str(time_natural)
    outfile = 'data/fitted_model_object{}.pkl'.format(fstub)
    pickle.dump(gpm, open(outfile, 'wb'))
