The `combined_data_1.txt`, `combined_data_2.txt`, `combined_data_3.txt`, and `combined_data_4.txt` files are required to be in this folder before `../dataparse.py` is executed.

These can be found in Kaggle's webpage for the Netflix competition.

After executing `../dataparse.py` the resulting files are:
`parseddf.parquet.gzip`: This is all of the movie rating records without the statistical outlier movies and users.
`reindexeddf.parquet.gzip`: This is the above file, but the movieid's are reindexed. This is necessary for setting up clustering with the PyClustering library.
`testusersdf.parquet.gzip`: This is the above file with 10 percent of users.
`trainingusersdf.parquet.gzip`: This is the reindexed dataframe with 90 percent of users.

The test set and training set dataframes are mutually exclusive. 
