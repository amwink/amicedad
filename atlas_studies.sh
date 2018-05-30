#!/bin/bash

CD=${PWD}

SL=${CD}/studies.txt
AL=${CD}/atlases.txt

for ATLAS in `cat ${AL}`;do
	
    for STUDY in `cat ${SL}`; do

        echo processing `basename ${STUDY}` with `basename ${ATLAS}`
        
        outdir=`basename ${STUDY}`
        outdir=${CD}/studies/${outdir%%_*}
        mkdir -p ${outdir}
        
        outfile=${outdir}/`basename ${STUDY%.*}`_x_`basename ${ATLAS%.*}`.txt        
        
        echo "writing in ${outdir} to file ${outfile}"

	SCAN1=`echo $(head -1 $STUDY)|awk '{print $1}'`
	study_pixdim=`fslinfo $SCAN1|grep pixdim1|awk '{print $2}'`
	atlas_pixdim=`fslinfo $ATLAS|grep pixdim1|awk '{print $2}'`	
	
	rm -rf /tmp/tmp_atlas* /tmp/tmp_mask* /tmp/tmp_subj

	if (( $(echo "$study_pixdim != $atlas_pixdim"|bc -l) )); then
	
	   cm="flirt -in $ATLAS -ref ${SCAN1} -out /tmp/tmp_atlas.nii.gz -applyxfm -interp nearestneighbour"
	   
	else
	   
	   cm="cp $ATLAS /tmp/tmp_atlas.nii.gz"
	
	fi

	echo $cm
	$cm

        printf "study pixdim: $study_pixdim, atlas pixdim: $atlas_pixdim, "
        echo "resampled atlas pixdim: `fslinfo /tmp/tmp_atlas|grep pixdim1`"

	CM="fslmeants -i ${ATLAS} --label=${ATLAS}"
	printf "$CM\r";$CM > ${outfile}
	rm -f ${outfile//.txt/_log.txt}

	cat ${STUDY} | awk '{print $1}' | while read SUBJ; do
	
	    cm1="fslmaths ${SUBJ} -nan /tmp/tmp_subj"
	    printf "$cm1\r"; 	    
	    $cm1	

	    cm2="fslmaths /tmp/tmp_subj -bin /tmp/tmp_mask"
	    printf "$cm2\r"; 
	    $cm2

	    cm3="fslmeants -i /tmp/tmp_subj -m /tmp/tmp_mask --label=/tmp/tmp_atlas"
	    printf "$cm3 \r";
	    echo "$cm1" >> ${outfile}
	    $cm3 >> ${outfile}
	    
	    rm -f /tmp/tmp_subj* /tmp/tmp_mask* 
	
	done	
	
	printf "\n"

	rm -rf /tmp/tmp_atlas* 

    done

done
