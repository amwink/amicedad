# coding: utf-8

# In[1]:

"""

script atlas_studies.py

This script reads in given outputs of image study analyses (eg GM density of ADNI), averaged inside given atlases (eg Harvard-Oxford)
and performs iterative feature selection using SVM (Wottschel 2017) to find the best classifier for each study/atlas pair.

(Wottschel 2017) -- discovery.ucl.ac.uk/1553224/1/Wottschel_thesis_final_VW.pdf

"""

import os
import math
import time
import itertools

import numpy as np
import pandas as pd
import sklearn as sl
import matplotlib.pyplot as plt

from datetime import date
from joblib import Parallel, delayed

from scipy.stats import ttest_ind
from sklearn import svm, grid_search
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import normalize, scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.cross_validation import StratifiedKFold, train_test_split, KFold, LeaveOneOut



def subj_lists( filename='adni_gm_3tp.txt',
                atlasdir='HO_cor_subcor_lat',
                atlasfile='HarvardOxford-sub_and_cort-maxprob-thr25-1mm.nii' ):
    
    # make and empty list for each patient group -- read as Pandas data frame
    df=pd.read_csv(filename,
                   names=['filename','diag','timepoint'],
                   header=None,
                   sep='\s+')

    # replace NaN (if only 1 tp) by 0
    df.fillna(0, inplace=True)
    
    # study - e.g. adni, geneva, cita or vumc followed by '_' and other things
    study=filename.split('_',1)[0]

    # fname - should be filename minus extension
    fname=filename.split('.',1)[0]

    # load all the regional values of study X with atlas Y
    na=np.loadtxt(os.path.join(os.path.dirname(filename),'studies',study,fname+'_x_'+atlasfile+'.txt'),skiprows=1)
    
    # put subjects (+ [timepoint and] diagnosis) together with regional values
    return(np.concatenate((df.values,na),axis=1))

def train_contrasts( rootdir='/home/amwink/work/analyses/memorabel_data_overview',
                    filename='adni_gm_3tp.txt',
                    join='python_atlas_svm',
                    atlasdir='HO_cor_subcor_lat',
                    atlasfile='HarvardOxford-sub_and_cort-maxprob-thr25-1mm.nii',
                    shrink=0.125,                    
                    use_nFold = 5,
                    use_nRepetition = 1000,
                    minFeatures = 7):

    # get the input data
    subjlists  = subj_lists( rootdir, filename, join, atlasdir, atlasfile )
    #print(subjlists)
    sl_nonames = subjlists[:,1:]
    
    # number of patient groups
    groups     = list(set(sl_nonames[:,0]))

    # number of time points
    timpts     = list(set(sl_nonames[:,1]))

    # number of possible contrasts
    contrasts  = list(itertools.combinations(groups,2))
    
    # study - should be adni, cita or vumc
    study=filename.split('_',1)[0]

    # fname - should be filename minus extension
    fname=filename.split('.',1)[0]

    for t in timpts:
        
        for c in contrasts:    
            
            # select the right subjects            
            sl_contrast0 = sl_nonames[ np.where ((sl_nonames[:,1]==t)*(sl_nonames[:,0]==c[0])) ]
            sl_contrast1 = sl_nonames[ np.where ((sl_nonames[:,1]==t)*(sl_nonames[:,0]==c[1])) ]

            # make y vector for svm
            y=np.concatenate( ([-1 for _ in range(len(sl_contrast0))],
                               [ 1 for _ in range(len(sl_contrast1))]) )

            # join contrast and remove timepoint / group labels
            sl_contrast=np.concatenate((sl_contrast0,sl_contrast1),axis=0)
            X=sl_contrast[:,2:]
            #print(X.shape)

            use_fold = 'stratified'
            input_downsample = True

            #try_C = (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100)
            try_C = (.01, .1, 1, 10)

            #this does not work on np.ndarray (it does on pd.dataframe)
            #X_all = (X - np.mean(X))/np.std(X)

            # instead, normalise with scale from sklearn along column direction
            X_all = scale(X, axis=0, with_mean=True, with_std=True, copy=True)

            useFeatures = np.arange(0,X_all.shape[1])
            RFE_acc = []
            RFE_bacc = []
            RFE_sens = []
            RFE_spec = []
            RFE_acc_min = []
            RFE_bacc_min = []
            RFE_sens_min = []
            RFE_spec_min = []
            RFE_acc_max = []
            RFE_bacc_max = []
            RFE_sens_max = []
            RFE_spec_max = []
            RFE_acc_train = []
            RFE_bacc_train = []
            RFE_sens_train = []
            RFE_spec_train = []
            RFE_acc_train_min = []
            RFE_bacc_train_min = []
            RFE_sens_train_min = []
            RFE_spec_train_min = []
            RFE_acc_train_max = []
            RFE_bacc_train_max = []
            RFE_sens_train_max = []
            RFE_spec_train_max = []

            RFE_pred = []
            RFE_pred_train = []
            RFE_groundTruth = []
            RFE_groundTruth_train = []
            RFE_nFeatures = []
            RFE_usedFeatures = []
            RFE_removedFeatures = []
            RFE_coeff_mean = []
            RFE_bias_mean = []

            k = 1

            # output filename
            outname_tmp="%s_x_%s_%d%d_t%d" % (filename, atlasfile, c[0], c[1], t)
            outname=os.path.join(rootdir,join,'studies',study,outname_tmp)
    
            while (len(useFeatures)>minFeatures):
                print('%d\r') % (len(useFeatures))              
                np.savez(outname+'_progress_'+str(k)+'.npz')
                k = k+1

                pred_perm = []
                pred_perm_train = []
                ground_truth_test_perm = []
                ground_truth_train_perm = []
                sens_perm = []
                spec_perm = []
                bacc_perm = []
                acc_perm = []
                sens_perm_train = []
                spec_perm_train = []
                bacc_perm_train = []
                acc_perm_train = []
                bacc_inner_perm = []
                acc_inner_perm = []
                sens_inner_perm = []
                spec_inner_perm = []
                bacc_inner_perm_train = []
                acc_inner_perm_train = []
                sens_inner_perm_train = []
                spec_inner_perm_train = []
                bacc_outer_perm = []
                acc_outer_perm = []
                sens_outer_perm = []
                spec_outer_perm = []
                bacc_outer_perm_train = []
                acc_outer_perm_train = []
                sens_outer_perm_train = []
                spec_outer_perm_train = []
                coef_outer_perm = []
                pred_outer_perm = []
                pred_outer_perm_train = []
                index_test_perm = []
                index_train_perm = []
                argmax_bacc_inner = []
                argmax_bacc_outer = []
                argmax_bacc_inner_train = []
                argmax_bacc_outer_train = []
                coef_fold = []
                bias_fold = []
                
                for repeat in np.arange(0, use_nRepetition):

                    print ('.'),
                    
                    X_all = (X - np.mean(X))/np.std(X)
                    y_all = y

                    if input_downsample == False:
                        X_cur = X_all[:,useFeatures]
                        y_cur = y_all
                    elif input_downsample == True:
                        y_1s = np.ravel(np.nonzero(y_all==1))
                        y_min1s = np.ravel(np.nonzero(y_all==-1))
                        if len(y_1s) > len(y_min1s):
                            sample = np.random.choice(y_1s, len(y_min1s), False)
                            indices = np.hstack((y_min1s, sample))
                        elif len(y_1s) < len(y_min1s):
                            sample = np.random.choice(y_min1s, len(y_1s), False)
                            indices = np.hstack((sample, y_1s))
                        else:
                            indices = np.hstack((y_min1s, y_1s))
                        X_cur = X_all[np.ix_(indices,useFeatures)]
                        y_cur = y_all[indices]

                    if use_fold == 'stratified':
                        kfolds = StratifiedKFold((y_cur), use_nFold, shuffle=True, random_state=None)
                    elif use_fold == 'kfold':
                        kfolds = KFold(len(y_cur), use_nFold, shuffle=True, random_state=None)
                    elif use_fold == 'loo':
                        kfolds = LeaveOneOut(X_cur.shape[0])

                    index_test_fold = []
                    index_train_fold = []
                    ground_truth_test_fold = []
                    ground_truth_train_fold = []

                    acc_fold = []
                    pred_fold = []
                    acc_fold_train = []
                    pred_fold_train = []
                    bacc_inner_fold = []
                    acc_inner_fold = []
                    sens_inner_fold = []
                    spec_inner_fold = []
                    bacc_inner_fold_train = []
                    acc_inner_fold_train = []
                    sens_inner_fold_train = []
                    spec_inner_fold_train = []
                    bacc_outer_fold = []
                    acc_outer_fold = []
                    sens_outer_fold = []
                    spec_outer_fold = []
                    pred_outer_fold = []
                    bacc_outer_fold_train = []
                    acc_outer_fold_train = []
                    sens_outer_fold_train = []
                    spec_outer_fold_train = []
                    pred_outer_fold_train = []
                    coef_outer_fold = []

                    for fold, (train_index, test_index) in enumerate(kfolds):

                        X_train, X_test = X_cur[train_index,:], X_cur[test_index,:]
                        y_train, y_test = y_cur[train_index], y_cur[test_index]

                        index_test_fold.append(test_index)
                        index_train_fold.append(train_index)

                        ground_truth_test_fold.append(y_test)
                        ground_truth_train_fold.append(y_train)

                        acc_inner_fold_tmp = []
                        sens_inner_fold_tmp = []
                        spec_inner_fold_tmp = []
                        acc_inner_fold_train_tmp = []
                        sens_inner_fold_train_tmp = []
                        spec_inner_fold_train_tmp = []

                        kfolds_inner = StratifiedKFold((y_train), 3, shuffle=True, random_state=None)
                        for fold_inner, (train_index_inner, test_index_inner) in enumerate(kfolds_inner):
                            X_train_inner, X_test_inner = X_train[train_index_inner,:], X_train[test_index_inner,:]
                            y_train_inner, y_test_inner = y_train[train_index_inner], y_train[test_index_inner]

                            acc_inner = np.zeros(len(try_C))
                            sens_inner = np.zeros(len(try_C))
                            spec_inner = np.zeros(len(try_C))
                            acc_inner_train = np.zeros(len(try_C))

                            for cur_C in np.arange(0,len(try_C)):

                                reg = LinearSVC(C=try_C[cur_C], class_weight='balanced')
                                reg.fit(X_train_inner, y_train_inner)
                                acc_inner[cur_C] = reg.score(X_test_inner, y_test_inner)
                                pred_inner = reg.predict(X_test_inner)
                                cm = confusion_matrix(y_test_inner, pred_inner)
                                if cm[0,0]==0:
                                    sens_inner[cur_C] = 0
                                else:
                                    sens_inner[cur_C] = float(cm[0,0])/(float(cm[0,0]) + float(cm[0,1]))
                                if cm[1,1]==0:
                                    spec_inner[cur_C] = 0
                                else:
                                    spec_inner[cur_C] = float(cm[1,1])/(float(cm[1,1]) + float(cm[1,0]))

                            acc_inner_fold_tmp.append(acc_inner)
                            sens_inner_fold_tmp.append(sens_inner)
                            spec_inner_fold_tmp.append(spec_inner)

                        bacc_inner_fold_tmp = (np.mean(sens_inner_fold_tmp, axis=0) + np.mean(spec_inner_fold_tmp, axis=0))/2
                        argmax_bacc_inner_tmp = np.argmax(bacc_inner_fold_tmp)
                        argmax_bacc_inner.append(argmax_bacc_inner_tmp)
                        acc_inner_fold.append(np.mean(acc_inner_fold_tmp, axis=0)[argmax_bacc_inner_tmp])

                        reg = LinearSVC(C=try_C[argmax_bacc_inner_tmp], class_weight='balanced')
                        reg.fit(X_train, y_train)
                        coef_fold.append(reg.coef_)
                        bias_fold.append(reg.intercept_)
                        acc_fold.append(reg.score(X_test, y_test))
                        pred_fold_tmp = reg.predict(X_test)
                        pred_fold.append(pred_fold_tmp)

                        acc_fold_train.append(reg.score(X_train, y_train))
                        pred_fold_train_tmp = reg.predict(X_train)
                        pred_fold_train.append(pred_fold_train_tmp)

                        #print '.',

                    pred_perm_tmp = np.concatenate(pred_fold)
                    pred_perm.append(pred_perm_tmp)
                    pred_perm_train_tmp = np.concatenate(pred_fold_train)
                    pred_perm_train.append(pred_perm_train_tmp)
                    ground_truth_test_perm_tmp = np.concatenate(ground_truth_test_fold)    
                    ground_truth_test_perm.append(ground_truth_test_perm_tmp)
                    ground_truth_train_perm_tmp = np.concatenate(ground_truth_train_fold)
                    ground_truth_train_perm.append(ground_truth_train_perm_tmp)
                    index_test_perm.append(np.concatenate(index_test_fold))
                    index_train_perm.append(np.concatenate(index_train_fold))
                    cm = confusion_matrix(ground_truth_test_perm_tmp, pred_perm_tmp)
                    if cm[0,0] == 0:
                        sens_perm_tmp = 0
                    else:
                        sens_perm_tmp = float(cm[0,0])/(float(cm[0,0]) + float(cm[0,1]))
                    if cm[1,1] == 0:
                        spec_perm_tmp = 0
                    else:
                        spec_perm_tmp = float(cm[1,1])/(float(cm[1,1]) + float(cm[1,0]))
                    sens_perm.append(sens_perm_tmp)
                    spec_perm.append(spec_perm_tmp)
                    bacc_perm.append((sens_perm_tmp + spec_perm_tmp)/2)
                    acc_perm.append((float(cm[0,0]) + float(cm[1,1]))/(float(cm[0,0]) + float(cm[1,1]) + float(cm[1,0]) + float(cm[0,1])))

                    cm_train = confusion_matrix(ground_truth_train_perm_tmp, pred_perm_train_tmp)
                    if cm_train[0,0] == 0:
                        sens_perm_train_tmp = 0
                    else:
                        sens_perm_train_tmp = float(cm_train[0,0])/(float(cm_train[0,0]) + float(cm_train[0,1]))
                    if cm_train[1,1] == 0:
                        spec_perm_train_tmp = 0
                    else:
                        spec_perm_train_tmp = float(cm_train[1,1])/(float(cm_train[1,1]) + float(cm_train[1,0]))
                    sens_perm_train.append(sens_perm_train_tmp)
                    spec_perm_train.append(spec_perm_train_tmp)
                    bacc_perm_train.append((sens_perm_train_tmp + spec_perm_train_tmp)/2)
                    acc_perm_train.append((float(cm_train[0,0]) + float(cm_train[1,1]))/(float(cm_train[0,0]) + float(cm_train[1,1]) + float(cm_train[1,0]) + float(cm_train[0,1])))

                coef_ranking = np.ravel(np.argsort(np.abs(np.sum((coef_fold), axis=0))))
                curXperc = int(np.round(useFeatures.size*shrink))
                worstXperc = useFeatures[coef_ranking[0:curXperc]]
                
                RFE_acc.append(np.mean(acc_perm))
                RFE_bacc.append(np.mean(bacc_perm))
                RFE_sens.append(np.mean(sens_perm))
                RFE_spec.append(np.mean(spec_perm))
                RFE_acc_min.append(np.min(acc_perm))
                RFE_bacc_min.append(np.min(bacc_perm))
                RFE_sens_min.append(np.min(sens_perm))
                RFE_spec_min.append(np.min(spec_perm))
                RFE_acc_max.append(np.max(acc_perm))
                RFE_bacc_max.append(np.max(bacc_perm))
                RFE_sens_max.append(np.max(sens_perm))
                RFE_spec_max.append(np.max(spec_perm))
                RFE_acc_train.append(np.mean(acc_perm_train))
                RFE_bacc_train.append(np.mean(bacc_perm_train))
                RFE_sens_train.append(np.mean(sens_perm_train))
                RFE_spec_train.append(np.mean(spec_perm_train))
                RFE_acc_train_min.append(np.min(acc_perm_train))
                RFE_bacc_train_min.append(np.min(bacc_perm_train))
                RFE_sens_train_min.append(np.min(sens_perm_train))
                RFE_spec_train_min.append(np.min(spec_perm_train))
                RFE_acc_train_max.append(np.max(acc_perm_train))
                RFE_bacc_train_max.append(np.max(bacc_perm_train))
                RFE_sens_train_max.append(np.max(sens_perm_train))
                RFE_spec_train_max.append(np.max(spec_perm_train))

                RFE_pred.append(pred_perm)
                RFE_pred_train.append(pred_perm_train)
                RFE_groundTruth.append(ground_truth_test_perm)
                RFE_groundTruth_train.append(ground_truth_train_perm)

                RFE_usedFeatures.append(useFeatures)
                RFE_nFeatures.append(useFeatures.size)
                RFE_removedFeatures.append(worstXperc)

                meanfold=np.ravel(np.mean((coef_fold), axis=0))
                meanbias=np.ravel(np.mean(bias_fold))
                RFE_coeff_mean.append(meanfold)
                RFE_bias_mean.append(meanbias)
               
                if np.size(RFE_acc) == 1:
                    remove = worstXperc
                    useFeatures = np.arange(0,X_all.shape[1])
                    useFeatures = np.delete(useFeatures, remove)
                else:
                    remove = np.hstack((remove, worstXperc))
                    useFeatures = np.arange(0,X_all.shape[1])
                    useFeatures = np.delete(useFeatures, remove)

            np.savez_compressed(outname+'.npz',
                                RFE_acc=RFE_acc, RFE_bacc=RFE_bacc, RFE_sens=RFE_sens, RFE_spec=RFE_spec,
                                RFE_acc_min=RFE_acc_min, RFE_bacc_min=RFE_bacc_min, RFE_sens_min=RFE_sens_min, RFE_spec_min=RFE_spec_min,
                                RFE_acc_max=RFE_acc_max, RFE_bacc_max=RFE_bacc_max, RFE_sens_max=RFE_sens_max, RFE_spec_max=RFE_spec_max,
                                RFE_acc_train=RFE_acc_train, RFE_bacc_train=RFE_bacc_train, RFE_sens_train=RFE_sens_train,             RFE_spec_train=RFE_spec_train,
                                RFE_acc_train_min=RFE_acc_train_min, RFE_bacc_train_min=RFE_bacc_train_min,
                                RFE_sens_train_min=RFE_sens_train_min, RFE_spec_train_min=RFE_spec_train_min, 
                                RFE_acc_train_max=RFE_acc_train_max, RFE_bacc_train_max=RFE_bacc_train_max, 
                                RFE_sens_train_max=RFE_sens_train_max, RFE_spec_train_max=RFE_spec_train_max,
                                RFE_pred=RFE_pred, RFE_pred_train=RFE_pred_train, 
                                RFE_groundTruth=RFE_groundTruth, RFE_groundTruth_train=RFE_groundTruth_train,
                                RFE_nFeatures=RFE_nFeatures, RFE_usedFeatures=RFE_usedFeatures, RFE_removedFeatures=RFE_removedFeatures,
                                RFE_coeff_mean=RFE_coeff_mean, RFE_bias_mean=RFE_bias_mean,
                                y=y, X_all=X_all )

            plt.cla()
            plt.plot(RFE_bacc)
            plt.plot(RFE_sens)
            plt.plot(RFE_spec)
            plt.legend(('bacc', 'sens', 'spec'), loc='lower right')
            plt.savefig(outname+'.png')


def process_pair(study, 
		 atlas, 
                 rootdir, 
		 join):
    
    stname=os.path.basename(study)
    
    atname=os.path.relpath(atlas,os.path.join(rootdir,join,'atlases'))
    atlasdir=atname.split('/',1)[0]
    atlasfile=atname.split('/',1)[1]

    print("%s, %s") % (study, atlasfile)
    train_contrasts ( rootdir,
                      stname,
                      join,
                      atlasdir,
                      atlasfile )

    
def main():

    # on the cloud
    # rootdir='/home/data/amwink/cloud_amwink/memorabel/memorabel_data_overview'
    # on the laptop
    rootdir='/home/amwink/work/analyses/memorabel_data_overview'
    join='python_atlas_svm'

    atlases = [line.rstrip('\n') for line in open(os.path.join(rootdir,join,'atlases.txt'))]
    #print(atlases)
    studies = [line.rstrip('\n') for line in open(os.path.join(rootdir,join,'studies_short.txt'))]
    #print(studies)

    Parallel(n_jobs=6)(delayed(process_pair)(s,a,rootdir,join) for s in studies for a in atlases)

    print("done!")
            
if __name__ == "__main__":
    main()
    # run with: >>> execfile ( "atlas_studies.py" )
