"""
Fit Gamma-Poisson topic model to movie/ethogram dataset.
"""
from __future__ import division
import numpy as np
import pandas as pd
import gamma_model as gp
import argparse
import cPickle as pickle

np.random.seed(12345)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run latent topic discovery using a Gamma-Poisson model")
    parser.add_argument("input", help="name of input file")

    args = parser.parse_args()

    # load up data
    datfile = args.input
    print "Reading data..."
    df = pd.read_csv(datfile)

    ######## for testing only ################
    print "Subsetting for testing..."
    df = df.query('time <= 1e5')
    # and renumber units consecutively (starting at 0)
    df['unit'] = np.unique(df['unit'], return_inverse=True)[1]
    ######## for testing only ################

    # set up params 
    print "Calculating parameters..."
    dt = 1. / 30  # duration of movie frame
    M = df.shape[0]
    T = df['time'].drop_duplicates().shape[0]
    U = df['unit'].drop_duplicates().shape[0]
    R = df.shape[1] - len(['time', 'unit', 'count'])
    K = 5

    # set up model object
    print "Initializing model..."
    gpm = gp.GammaModel(df, K)

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
    fr_shape_rate = 1e-3 * np.ones((K,))
    fr_mean_shape = 0.4 * T * np.ones((K,))
    fr_mean_rate = 0.4 * T * np.ones((K,))

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
                'post_child_rate': np.ones((U, K))
                })

    ############ latent states ####################
    ###### A ###############
    A_off = 10.
    A_on = 1.
    Avec = np.r_[A_off, A_on].reshape(2, 1, 1)
    A_prior = np.tile(Avec, (1, 2, K))

    ###### pi ###############
    pi_off = 15.
    pi_on = 1.
    pi_prior = np.tile(np.r_[pi_off, pi_on].reshape(2, 1), (1, K))

    # E[z]
    # initialize pretty much at random (10% 1's)
    rand_frac = 0.1
    xi_mat = (rand_frac >= np.random.rand(T, K))
    xi_mat = xi_mat.astype('float')
    z_prior = np.dstack([1 - xi_mat, xi_mat]).transpose((2, 0, 1))

    # E[zz]
    Xi_mat = np.random.rand(2, 2, T - 1, K)
    Xi_mat = Xi_mat.astype('float')

    latent_dict = ({'A_prior': A_prior, 'pi_prior': pi_prior,
                    'A_post': A_prior, 'pi_post': pi_prior, 
                    'z_init': z_prior, 'zz_init': Xi_mat, 
                    'logZ_init': np.zeros((K,))
                    })

    ############ regression coefficients ####################
    ups_shape = 11.
    ups_rate = 10.
    reg_shape = ups_shape * np.ones((U, R))  # shape
    reg_rate = ups_rate * np.ones((U, R))  # rate

    # since we know exact update for a_mat, use that
    nn = df['count']
    uu = df['unit']
    NX = nn[:, np.newaxis] * df.iloc[:, -R:]
    NX *= 0.5  # don't start off too big
    a_mat = NX.groupby(uu).sum().values + reg_rate
    b_mat = a_mat.copy()

    reg_dict = ({'prior_shape': reg_shape, 'prior_rate': reg_rate,
                'post_shape': a_mat, 'post_rate': b_mat
                 })

    ############ overdispersion ####################
    od_shape = 6.
    od_rate = 5.
    od_dict = ({'prior_shape': od_shape * np.ones((M,)), 
                'prior_rate': od_rate * np.ones((M,)), 
                'post_shape': np.ones((M,)), 
                'post_rate': np.ones((M,))
                })

    ############ initialize model ####################
    gpm.initialize_baseline(**baseline_dict)
    gpm.initialize_fr_latents(**fr_latent_dict)
    gpm.initialize_latents(**latent_dict)
    gpm.initialize_fr_regressors(**reg_dict)
    gpm.initialize_overdispersion(**od_dict)
    gpm.finalize()

    ############## fit model
    print "Fitting model..."
    gpm.do_inference(tol=1e-3, verbosity=2)

    print "Cleaning up..."
    # need to get rid of externally defined functions for pickling
    # also has the effect of "neutering" object from futher update
    for name, node in gpm.nodes.iteritems():
        attrs = node.__dict__.keys()
        to_delete = [a for a in attrs if 'update' in a]
        for a in to_delete:
            delattr(node, a)

    print "Writing to disk..."
    outfile = 'data/fitted_model_object.pkl'
    pickle.dump(gpm, open(outfile, 'wb'))
