spiketopics
===========

Inferring binary features for neural populations.

This code implements a version of the Gamma-Poisson model on a pseudopopulation of independently recorded neurons. Details of the model and inference are in `gamma_poisson_notes.tex`.

## Model fitting:

### Integration tests:
- `gamma_poisson_scratch.ipynb`: Uses the Gamma-Poisson model to fit synthetic data and successfully infer the underlying Markov chain.
- `gpm_sanity_check.ipynb`: Another integration test of inference in the Gamma-Poisson model, this time with hierarchical priors on firing rates for each feature to induce parsimony.
- `lognormal_model_integration_test.ipynb`: Similar to the integration tests above, but using a semi-Markov model defined by a lognormal prior on dwell-time in each state.

### Model Code
- `gamma_model.py`: Implements the Gamma model for firing rates with HMM state space dynamics.
- `lognormal_model.py`: Implements the LogNormal model with HSMM state space dynamics.
- `gamma_poisson.py`: Old (deprecated?) version of `gamma_model.py`
- `helpers.py`: miscellaneous functions useful for setting up the model and evaulating outputs.

#### Markov inference
- `fbi.py`: Forward-Backward algorithm implemented in pure Python.
- `forward_backward.py`: Forward-Backward algorithm implemented using Numba's JIT facilities to provide much faster inference.
- `hsmm_forward_backward.py`: Forward-Backward inference for the hidden semi-Markov model. Again uses Numba to provide just-in-time compilation.

## Nodes:
The algorithm is defined by a graphical model with nodes in a directed graph corresponding to each variable. The `nodes` folder contains code defining odes for several common distribution types.

## Testing:
Unit tests are located in the `tests` folder. These can be run in their entirety by
~~~
nosetests
~~~
or as modules by
~~~
nosetests tests/name_of_test_module.py
~~~

## Case studies:
The `experiments` folder contains case studies for the application of the algorithm to test data sets.

## Documentation:
The `docs` folder contains notes, papers, and other documentation for the algorithm.

## TODO:
[ ] Correlated noise across neurons (log-Normal model does this)
[ ] Response sparsity within populations, not just features
