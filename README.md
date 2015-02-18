spiketopics
===========

Inferring binary features for neural populations.

This code implements a version of the Gamma-Poisson model on a pseudopopulation of independently recorded neurons. Details of the model and inference are in `gamma_poisson_notes.tex`.

## Data preparation and extraction
- The `sql` folder contains scripts needed to extract and process data from the SQL database.
- it is assumed the MySQL server is running on the local machine
- `get_all_data.sh` is the master script it calls:
    - `extract_spike_data.py`, which pulls spike data from the database and saves a csv of spike counts for each (neuron, frame)
    - `get_etho.sh`, which pulls the ethogram from the database
    - `get_presentations.sh`, which pulls the information on what clips were presented to what neurons
    - `merge_counts_with_presentations.py`, which bundles all the training data up into a single csv file
    - these scripts all call associated `.sql` files for their particular queries

## Model fitting:
- `gamma_poisson.py` is a module containing an implementation of the Forward-Backward algorithm and a class definition for the `GPModel` class
- `discover_topics.py` reads data from a csv file, initializes the model, runs inference, and writes the results to disk
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

   

