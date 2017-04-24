#!/bin/bash

src=~/Dropbox/UdeS/ProgCD/Python/Python3/Physique/Mea2_0Refactor/auxiliary_fn

if [ -d ~/storage ]
then
        echo "Directory exists"
	dest=~/storage/TEST_long
else
        echo "Directory does not Exists"
	dest=~/Desktop/TEST_long
	
fi

python_path=/home/charles-david/Documents/Installations/Anaconda3/bin/
OmegaMaxEnt_path=/home/charles-david/Documents/Installations/OmegaMaxEnt_source
export PATH="$python_path:$OmegaMaxEnt_path:$PATH"
echo $PATH
#rm -r $dest/*

for i in `seq 1 3`;
do
#echo
#	rm -r $dest/* 2> /dev/null
	rsync -avzq --delete $src $dest
	cd $dest/auxiliary_fn
	python -m unittest discover
	python tests/long_test_kramerskronig.py
#	python tests/test_green.py
#	python tests/test_acon.py
#	python tests/test_auxiliary_fn.py
	( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.1s ; kill -9 $pid
done
