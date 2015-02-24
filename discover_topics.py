"""
Fit Gamma-Poisson topic model to movie/ethogram dataset.
"""
from __future__ import division
import numpy as np
import pandas as pd
import gamma_poisson as gp

np.random.seed(12345)

# load up data
datfile = './sql/spike_presentations.csv'

print "Reading data..."
df = pd.read_csv(datfile)

print "Processing data..."
df = df.drop(['trialId', 'frameClipNumber'], axis=1)
df = df.sort(['movieId', 'frameNumber', 'unitId'])
df = df.rename(columns={'unitId': 'unit', 'frameNumber': 'frame', 
    'movieId': 'movie'})

######## for testing only ################
# df = df.query('movie <= 50')
# # and renumber units consecutively
# df['unit'] = np.unique(df['unit'], return_inverse=True)[1]
######## for testing only ################

# set up params 
print "Calculating parameters..."
dt = 1 / 30  # duration of movie frame
T = df[['movie', 'frame']].drop_duplicates().shape[0]
U = df['unit'].drop_duplicates().shape[0]
K = 5

# set up model object
print "Initializing model..."
gpm = gp.GPModel(T, K, U, dt)
gpm = gpm.set_data(df)

#################### priors and initial values

print "Setting priors and inits..."
# multipliers shouldn't be much bigger than ~4
cmat = 2 * np.ones((K, U))
dmat = 2 * np.ones((K, U))

# baselines can be a lot bigger
cmat[0, :] = 3
dmat[0, :] = 1/6 / dt  # only baseline needs to compensate for dt

nu1_mat = np.r_[np.ones((1, K)), 10 * np.ones((1, K))]
nu2_mat = np.r_[10 * np.ones((1, K)), np.ones((1, K))]

rho1_mat = np.ones((K,))
rho2_mat = np.ones((K,))

alpha_mat = (np.sum(gpm.N, axis = 0) + 1 + cmat).data

# start by assuming beta very large
beta_mat = dmat + 100000

# for an initial guess for A's posterior params, just take the the prior
gamma1_mat = nu1_mat
gamma2_mat = nu2_mat

# again, just take the prior
delta1_mat = rho1_mat
delta2_mat = rho2_mat

#xi_mat = np.round(0.1 * np.random.rand(T, K))
xi_mat = np.zeros((T, K))
xi_mat[:, 0] = 1  # baseline

Xi = np.random.rand(T - 1, K, 2, 2)

# and two-slice marginal for baseline
Xi[:, 0] = 0
Xi[:, 0, 1, 1] = 1
priors = {'cc': cmat, 'dd': dmat, 'nu1': nu1_mat, 'nu2': nu2_mat,
          'rho1': rho1_mat, 'rho2': rho2_mat}

inits = {'alpha': alpha_mat, 'beta': beta_mat, 
         'gamma1': gamma1_mat, 'gamma2': gamma2_mat,
         'delta1': delta1_mat, 'delta2': delta2_mat,
         'xi': xi_mat, 'Xi': Xi}

gpm.set_priors(**priors).set_inits(**inits)

############## fit model
print "Fitting model..."
gpm.do_inference(silent=False, tol=5e-3)

print "Writing to disk..."
ethoframe = pd.concat([gpm.t_index, pd.DataFrame(gpm.xi)], axis=1)
ethoframe.to_csv('inferred_etho.csv', index=False)