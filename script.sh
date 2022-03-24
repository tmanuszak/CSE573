#!/bin/bash

if [ ! -f ./Data/cluster_*.txt ]; then
	python3 cluster.py
fi

if [ ! -f ./Data/results.txt ]; then
	python3 classify.py
fi
