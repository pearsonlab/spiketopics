# spiketopics
===========

Inferring binary features for neural populations.

This code implements a version of the Gamma-Poisson model on a pseudopopulation of independently recorded neurons. Details of the model and inference are in `gamma_poisson_notes.tex`.

## Model fitting:

### Model Code
- `gamma_model.py`: Implements the Gamma model for firing rates with HMM state space dynamics.
- `helpers.py`: miscellaneous functions useful for setting up the model and evaulating outputs.

#### Markov inference
- `forward_backward.py`: Forward-Backward algorithm implemented using Numba's JIT facilities to provide much faster inference.
- `hsmm_forward_backward.py`: Forward-Backward inference for the hidden semi-Markov model. Again uses Numba to provide just-in-time compilation.


## Archived
Deprecated files from semi-Markov model in gamma poisson model and lognormal model. The `archive` folder contains the codes that are not used in our current settings, but kept for recording purposes. 


## Documentation:
The `docs` folder contains notes, papers, and other documentation for the algorithms.


## Case Studies
The `experiments` folder contains case studies for the application of the algorithm to test data sets.


## Nodes
The algorithm is defined by a graphical model with nodes in a directed graph corresponding to each variable. The `nodes` folder contains code defining odes for several common distribution types.


## Testing
Unit tests are located in the `tests` folder. These can be run in their entirety by
~~~
nosetests
~~~
or as modules by
~~~
nosetests tests/name_of_test_module.py
~~~



## TODO:
- [ ] Correlated noise across neurons (log-Normal model does this)
- [x] Response sparsity within populations, not just features
- [ ] Smarter feature sparsity
- [x] Noise autocorrelation in time (hard)
