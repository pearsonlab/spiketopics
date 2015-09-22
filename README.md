spiketopics
===========

Inferring binary features for neural populations.

This code implements a version of the Gamma-Poisson model on a pseudopopulation of independently recorded neurons. Details of the model and inference are in `gamma_poisson_notes.tex`.

## Model fitting:
- `gamma_poisson.py` is a module containing an implementation of the Forward-Backward algorithm and a class definition for the `GPModel` class
- There are also several IPython notebooks:
    - `gpm_sanity_check` and `gamma_poisson_scratch` fit the model to synthetic data generated from the model. These illustrate both the underlying generative model and the steps used to set up inference.
    - `explore_inferred_etho` loads the results of model fitting and attempts to correlate inferred binary features with those in the hand-coded ethogram.

## Testing:
Unit tests are located in the `tests` folder. These can be run in their entirety by
~~~
nosetests
~~~
or as modules by
~~~
nosetests tests/name_of_test_module.py
~~~



