import os;
import glob;
import numpy as np;
import nibabel as nib;
import matplotlib as plt;

from nilearn.plotting import plot_glass_brain,show;

npzfiles=glob.glob('*/*/*.npz')                 # list of npz files (N=96)

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

for npzfile in npzfiles:
    if not npzfile.count('progress'):

        # print('loading the data'),
        npzdata=np.load(npzfile)                  # load the npz contents
        for varName in npzdata:                   # this is allowed to trawl npz variables???
            locals()[varName]=npzdata[varName]   # just set each of them as a global for now
            #print(vars().keys())                 # show all variable names - as said, always the same
  
        # print('determining the optimal parameters'),
        acc_opt=np.max(RFE_bacc)
        feat_opt=RFE_usedFeatures[np.argmax(RFE_bacc)];
        coeff_opt=RFE_coeff_mean[np.argmax(RFE_bacc)];
  
        print("%-99s\t %06.4f") % (npzfile+':',acc_opt); # show the name (vars are always named the same)
