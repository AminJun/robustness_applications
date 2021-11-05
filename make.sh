#!/bin/bash 
for l in 1.0 3.0 10.0 30.0 100. ; do 
	for m in `seq 0 6 `; do 
		echo "python inv_1000.py -m ${m} -l ${l}"
	done
done
