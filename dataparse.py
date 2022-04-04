import pandas as pd
import numpy as np
import os
import fastparquet

# If we dont already have a dataframe of the parsed ratings data
if not os.path.isfile('./Data/df.parquet.gzip'):
    if not os.path.isfile('./Data/data.csv'):

        # read all txt file and store them in one big file
        data = open('./Data/data.csv', mode='w')
        row = list()
        files = ['./Data/combined_data_1.txt', './Data/combined_data_2.txt',
                 './Data/combined_data_3.txt', './Data/combined_data_4.txt']
        for file in files:
            with open(file) as f:
                for line in f:
                    del row[:]
                    line = line.strip()
                    if line.endswith(':'):
                        movid_id = line.replace(':', '')
                    else:
                        row = [x for x in line.split(',')]
                        row.insert(0, movid_id)
                        data.write(','.join(row))
                        data.write('\n')
            print('Done.\n')
        data.close()

    df = pd.read_csv('./Data/data.csv', sep=',', names=['MovieId', 'UserId', 'Rating'], usecols=[0, 1, 2])

    # Parsing out the outliers
    df = df[df.groupby('UserId').transform(len)['Rating'] > 19]  # Remove bottom 10% of UserIds
    df = df[df.groupby('UserId').transform(len)['Rating'] < 1390]  # Remove top 1% of UserIds
    df = df[df.groupby('MovieId').transform(len)['Rating'] > 85]  # Remove bottom 10% of MovieIds

    # Make a parquet file of the dataframe so it is quick to load for later use
    df.to_parquet('./Data/df.parquet.gzip', compression='gzip')

    # This is much larger than the parquet file and isnt parsed for outliers. No longer needed.
    os.remove("./Data/data.csv")


# Read the dataframe of the parsed ratings data
df = pd.read_parquet('./Data/df.parquet.gzip')

# The MovieId needs to be reindexed [1 ... max] to easily cluster
unique_movieid_list = sorted(df['MovieId'].unique())
new_movieids = [0] * unique_movieid_list[-1]

# Generate the test set. This is 10% of users picked deterministically. The statistical attributes of this set is the
# same as the training set.
if not os.path.isfile('./Data/testusersdf.parquet.gzip'):
    df2 = df.loc[df['UserId'] % 10 == 0]
    df2.to_parquet('./Data/testusersdf.parquet.gzip', compression='gzip')


# Generate the training set. This is 90% of users picked deterministically. The statistical attributes of this set is
# the same as the test set.
if not os.path.isfile('./Data/trainingusersdf.parquet.gzip'):
    df2 = df.loc[df['UserId'] % 10 != 0]
    df2.to_parquet('./Data/trainingusersdf.parquet.gzip', compression='gzip')

# Generate csv of training data with filename "./Data/traininglist.csv" for the PyClustering library to use
# The file will have the following format (all values are rating values):
#       movie1user1, movie2user1, movie3user1, ... , movieMuser1
#       movie1user2, movie2user2, movie3user2, ... , movieMuser2
#            ...         ...         ...        ...      ...
#       movie1userU, movie2userU, movie3userU, ... , movieMuserU
#
# Here movie1user1 is the Rating that UserId=1 gave MovieId=1. Here M is the movie with the largest id and U is the
# user with the largest id. If a user has not rated a movie, then it is 0.