# Inferring Ethogram Categories
This folder contains code needed for running the inference code on the neural data from Adams et al. Files are described below:

# SQL
This folder contains code used to extract the spike counts data from the MySQL database.

## Data extraction and prep
- `get_all_data`: high-level bash script that calls
    - `extract_spike_data.py`: extract spike data from the database and save as csv
    - `get_etho: bash script containing SQL query to grab the hand-labeled ethogram categories from the database
    - `get_presentations`: bash script that extracts the table of presentations (which movie clip was played each trial) based on `get_presentations.sql`.
    - `merge_counts_with_presentatsions.py`: python code for merging spike data with data on movie clip presentations
    - The result of all these is a csv file containing one row for each data point (recorded unit, trial, time in clip, spike count, frame in movie, and movie).

## Other files:

### Notebooks
- `db_extraction_scratch.ipynb`: gist for connecting to SQL server and extracting data using sqlalchemy and pandas packages.
-  `extract_etho_data.ipynb`: same idea, but for ethogram data

### SQL files
- `get_presentations.sql`: currently the only file called in data extraction
- `get_unit_run.sql`, `get_spikes_for_unit_run.sql`, `get_all_runs_for_units.sql`: these appear to have been subsumed and hard-coded into `extract_spike_data.py`