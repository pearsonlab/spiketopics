import pandas as pd

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