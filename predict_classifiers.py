import os;
import glob;
import itertools;
import numpy as np;
import nibabel as nib;
import matplotlib.pyplot as plt;

npzfiles=glob.glob('*/*/*.npz');
npzfiles=sorted(npzfiles);

print(npzfiles)

for npzfile in npzfiles:
    atlstring=npzfile.split('_x_')[1].split('.nii')[0]+'.nii';
    npzdata=np.load(npzfile);

    for varname in ('RFE_bacc','RFE_usedFeatures','RFE_removedFeatures','RFE_coeff_mean','RFE_bias_mean'):
        locals()[varname]=npzdata[varname];
    #print(locals().keys());

    acc_opt=np.max(RFE_bacc);
    print('optimal accuracy: ' + str(acc_opt));

    feat_opt=RFE_usedFeatures[np.argmax(RFE_bacc)];
    print('feature set: ' + str(feat_opt));

    feat_rem=np.concatenate(RFE_removedFeatures[:(np.argmax(RFE_bacc))]);
    coeff_opt=RFE_coeff_mean[np.argmax(RFE_bacc)];
    print('removed features: ' + str(feat_rem));

    print('total #features: ' + str (len(feat_opt)+len(feat_rem)) )
    
    bias_opt=RFE_bias_mean[np.argmax(RFE_bacc)];
    print('bias term: ' + str(bias_opt));
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    predictions_file=os.path.join(os.path.dirname(npzfile),"predictions_"+os.path.basename(npzfile).split(".npz",1)[0]+".txt");
    print("predictions file name: "+predictions_file);
    file=open(predictions_file,"w")

    for npzfile2 in npzfiles:
        if not npzfile2.count('progress'):
            if ( npzfile2.count(atlstring) and (npzfile2.count("_plus_") == npzfile.count("_plus_")) ):
                
                npzdata2=np.load(npzfile2);
                for varName in ('y','X_all'):                 
                    locals()[varName]=npzdata2[varName];
            
                # variables for ROC
                y2=y/2+1;
                fpr = dict();
                tpr = dict();
                roc_auc = dict();
               
                #print(y.shape);
                #print(X_all.shape);
                cls=np.zeros(y.shape);

                for i in range(y.shape[0]):
                    coeff_sel=(X_all[i,feat_opt])
                    cls[i]=sum([m*n for m,n in zip(coeff_sel,coeff_opt)])+bias_opt;
                    #print('y2={},\tcls={}'.format(y2[i],cls[i]))
        
                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(y2, cls);
                    roc_auc[i] = auc(fpr[i], tpr[i]);
            
            
                #print("%-80s\t") % npzfile2, 
                #print roc_auc_score(y2, cls)
                file.write("%-80s -> %-80s\t%8.4f\n" % (npzfile, npzfile2, roc_auc_score(y2, cls)))
            
                #plt.figure()
                #plt.plot(fpr[1], tpr[1])
                #plt.xlim([0.0, 1.0])
                #plt.ylim([0.0, 1.05])
                #plt.xlabel('False Positive Rate')
                #plt.ylabel('True Positive Rate')
                #plt.title('Receiver operating characteristic')
                #plt.show()
            
    file.close();
