#!/bin/bash

CD=$PWD

if [[ $1 == "" ]]; then
    ATLASES=$(ls studies/*/prediction*txt | sed -e 's/.*_x_//' | sed -e 's/.nii.*/.nii/' | sort | uniq)
    printf "give atlas: "
    echo $ATLASES
    exit 1
fi

ATLAS=$1




NOPLUS=$(ls -1 studies/*/prediction*${ATLAS}* | grep -v _plus_ | sort -k 3)
NOPLUS=$(echo "$NOPLUS"|head -1)
NOPLUS=$(awk '{print $3}' $NOPLUS)
NOPLUS=${NOPLUS//.npz/.txt}
NOPLS2=""
for l in $NOPLUS; do
    NOPLS2=$(printf "%s\n%s" "$NOPLS2" $(dirname $l)/predictions_$(basename $l))
done

MATRIX=""
for F in $NOPLS2; do

    COLUMN="$(awk '{print $4}' $F)"
    
    MATRIX="$(paste <(echo "$MATRIX") <(echo "$COLUMN") --delimiters ' \t')"
    
done
paste <(echo "$NOPLS2") <(echo "$MATRIX") --delimiters ' ' > ${ATLAS}_noplus.txt




DOPLUS=$(ls -1 studies/*/prediction*_plus_*${ATLAS}* | sort -k 3)
DOPLUS=$(echo "$DOPLUS"|head -1)
DOPLUS=$(awk '{print $3}' $DOPLUS)
DOPLUS=${DOPLUS//.npz/.txt}
DOPLS2=""
for l in $DOPLUS; do
    DOPLS2=$(printf "%s\n%s" "$DOPLS2" $(dirname $l)/predictions_$(basename $l))
done

MATRIX=""
for F in $DOPLS2; do

    COLUMN="$(awk '{print $4}' $F)"
    
    MATRIX="$(paste <(echo "$MATRIX") <(echo "$COLUMN") --delimiters ' ')"
    
done
paste <(echo "$DOPLS2") <(echo "$MATRIX") --delimiters ' \t' > ${ATLAS}_doplus.txt



echo "#################################"
echo "$DOPLS2"
echo "#################################"

exit 0
