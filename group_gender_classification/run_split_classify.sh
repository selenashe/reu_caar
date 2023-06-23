#!/bin/sh

declare -a arr = (list of files)

cd $working_dir

for file in ./*;do
	python3 split_and_classify.py #with some sort of flags to make it run easier
done


