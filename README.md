Put the combined\_data\_\*.txt files in the Data folder.

`cluster.py` parses the data, defines a `Data/trainingset.txt` and `Data/testset.txt`, and generates the clusters. Output is written to `Data/clusters_kmeans.txt.txt` and `Data/clusters_graph.txt`.

`classify.py` tests the testing data and outputs the results/stats to `Data/results_kmeans.txt` and `Data/results_graph.txt`.
