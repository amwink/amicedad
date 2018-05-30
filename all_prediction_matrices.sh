#!/bin/bash

for f in `echo $(prediction_matrix.sh)|awk '{print $3,$4,$5}'`;do

    cm="prediction_matrix.sh $f";
    echo $cm;
    $cm;

done
