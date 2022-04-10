import pandas as pd
import numpy as np
import os
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Loading the ratings data and filtering out the outliers
if not os.path.isfile('./Data/removedoutliersdf.parquet.gzip'):
    # import the rating data as pandas dataframe
    df = pd.read_csv('./Data/combined_data_1.txt', header=None, names=['UserId', 'Rating'], usecols=[0, 1])

    df['Rating'] = df['Rating'].astype(float)  # Rating is temporarily a float

    df.index = np.arange(0, len(df))  # reindex the ratings

    # Adding the MovieId to the data frame
    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        # numpy approach
        temp = np.full((1, i-j-1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    df = df[pd.notnull(df['Rating'])]

    df['MovieId'] = movie_np.astype(int)
    df['UserId'] = df['UserId'].astype(int)


    # Removing unpopular movies and users with too few reviews
    # Removing the 70% least popular movies and users with the least ratings
    f = ['count', 'mean']

    df_movie_summary = df.groupby('MovieId')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    print('Movie minimum times of review: {}'.format(movie_benchmark))

    df_cust_summary = df.groupby('UserId')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    print('Customer minimum times of review: {}'.format(cust_benchmark))

    print('Original Shape: {}'.format(df.shape))
    df = df[~df['MovieId'].isin(drop_movie_list)]
    df = df[~df['UserId'].isin(drop_cust_list)]

    df['Rating'] = df['Rating'].astype(int)

    print('After Trim Shape: {}'.format(df.shape))

    print(df.describe())

    df.to_parquet('./Data/removedoutliersdf.parquet.gzip', compression='gzip')


# Get a list of movie titles that passed the above filter
if not os.path.isfile('./Data/filteredmovietitlesdf.parquet.gzip'):
    df = pd.read_parquet('./Data/removedoutliersdf.parquet.gzip')
    df_title = pd.read_csv('./Data/movie_titles.csv', encoding="ISO-8859-1", header=None, usecols=[0, 2],
                           names=['MovieId', 'Name'])
    # df_title.set_index('MovieId', inplace=True)
    df_title = pd.merge(df, df_title, how='inner', on='MovieId').drop_duplicates(subset=['MovieId'])[['MovieId', 'Name']]
    df_title.to_parquet('./Data/filteredmovietitlesdf.parquet.gzip', compression='gzip')

# Make a training set of users
if not os.path.isfile('./Data/trainingusersdf.parquet.gzip'):
    df = pd.read_parquet('./Data/removedoutliersdf.parquet.gzip')
    df2 = df.loc[df['UserId'] % 10 != 0]
    print("Original data set statistics:")
    print(df.describe())
    print("Training data set statistics:")
    print(df2.describe())
    df2.to_parquet('./Data/trainingusersdf.parquet.gzip', compression='gzip')

# Make a test set of users
if not os.path.isfile('./Data/testusersdf.parquet.gzip'):
    df = pd.read_parquet('./Data/removedoutliersdf.parquet.gzip')
    df2 = df.loc[df['UserId'] % 10 == 0]
    print("Original data set statistics:")
    print(df.describe())
    print("Test data set statistics:")
    print(df2.describe())
    df2.to_parquet('./Data/testusersdf.parquet.gzip', compression='gzip')

# Pivot the data frame into a user-item matrix
df = pd.read_parquet('./Data/trainingusersdf.parquet.gzip')
df = pd.pivot_table(df, values='Rating', index='UserId', columns='MovieId', fill_value=0)

# Finding best number of clusters for kmeans
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=[i for i in range(18,26)], timings=True)
visualizer.fit(df)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure