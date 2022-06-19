#!/bin/bash

# delete invalid files
find . -name "*.jpg" -type 'f' -size -2k -delete

# generate files list
files=$(find . -name "*.jpg" -type 'f')

# generate labels and write gt file
for i in $files
do
    echo $i
    pic_date=$(echo $i | grep -Po "[0-9]{4}(?=\.)")
    dob=$(echo $i | grep -Po "(?<=_)[0-9]{4}(?=\-)")
    [ -z "$pic_date" ] && continue
    [ -z "$dob" ] && continue
    class=$(($pic_date - $dob))
    echo $class,$i >> gt.csv
done

sed -i '1s/^/label,image\n/' gt.csv


# python stuff
# python prepare_age_dataset.py