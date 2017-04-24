#!/bin/bash

src=~/Dropbox/UdeS/ProgCD/Python/Python3/Physique/Mea2_0Refactor/auxiliary_fn

if [ -d $src ]
then
	echo "src is Dropbox"
else
	echo "src is OneDrive"
	src="/mnt/e/OneDrive - USherbrooke/CDH/UdeS/ProgCD/Python/Python3/Physique/Mea2_0Refactor/auxiliary_fn"
fi

if [ -d ~/storage ]
then
        echo "Directory exists"
	dest=~/storage/TEST_short
else
        echo "Directory does not Exists"
	dest=~/Desktop/TEST_short
	
fi


rsync -avzq --delete "$src" "$dest"
cd "$dest/auxiliary_fn"
python -m unittest discover
