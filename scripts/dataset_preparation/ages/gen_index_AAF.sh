#!/bin/bash

# generate files list
files=$(find './original images/' -name "*.jpg" -type 'f')

# generate labels and write gt file
IFS=$'\n'  # fix bash thinking it's a good idea to not online split string by \n but also by spaces
for i in $files
do
    echo "$i"
    class=$(echo $i | grep -Po "(?<=A)[0-9]{2}")
    echo $class,$i >> gt.csv
done

sed -i '1s/^/label,image\n/' gt.csv


# python stuff
python prepare_age_dataset.py
