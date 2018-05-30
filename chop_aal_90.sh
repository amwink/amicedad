#!/bin/bash

CD=${PWD}

for f in `find ${CD}/studies -name \*aal_\*mm.nii.txt`;do
    echo $f
    cat $f|wc
    CM="cut -f1-90 -d' ' $f > ${f//.nii/_90.nii}"
    echo $CM
    cut -f1-90 -d' ' $f > ${f//.nii/_90.nii}
done
