## Archieved

Deprecated files from semi-Markov model in gamma poisson model and lognormal model. The `archieve` folder contains the codes that are not used in our current settings, but kept for recording purposes. 

### Integration tests:

- `gamma_poisson_scratch.ipynb`: Uses the Gamma-Poisson model to fit synthetic data and successfully infer the underlying Markov chain.
- `gpm_sanity_check.ipynb`: Another integration test of inference in the Gamma-Poisson model, this time with hierarchical priors on firing rates for each feature to induce parsimony.
- `lognormal_model_integration_test.ipynb`: Similar to the integration tests above, but using a semi-Markov model defined by a lognormal prior on dwell-time in each state.

### Model Code

- `lognormal_model.py`: Implements the LogNormal model with HSMM state space dynamics.
- `gamma_poisson.py`: Old (deprecated?) version of `gamma_model.py`


### Markov inference
- `fbi.py`: Forward-Backward algorithm implemented in pure Python.