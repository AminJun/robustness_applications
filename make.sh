#!/bin/bash 
for l in 0.1 1.0 3.0 10.0; do 
	for m in `seq 0 6 `; do 
		echo "python inv10.py -m ${m} -l ${l}"
	done
done
