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
        #print(npzfile)                         # show the name (vars are always named the same)

        print('loading the data'),
        npzdata=np.load(npzfile)                # load the npz contents
        for varName in npzdata:                 # this is allowed to trawl npz variables???
            locals()[varName]=npzdata[varName] # just set each of them as a global for now
            #print(vars().keys())               # show all variable names - as said, always the same

        print('determining the optimal parameters'),
        acc_opt=np.max(RFE_bacc)
        feat_opt=RFE_usedFeatures[np.argmax(RFE_bacc)];
        coeff_opt=RFE_coeff_mean[np.argmax(RFE_bacc)];

        print('setting the paths'),
        currentdir=os.path.abspath(os.path.curdir);
        rootdir=os.path.dirname(currentdir);
        join=os.path.basename(currentdir);
        atlasfile=npzfile.split('_x_')[1].split('.nii')[0]+'.nii';
        
        #atlaspath=glob.glob(os.path.join(rootdir,join,'atlases')+'/*/'+atlasfile);
        atlaspath=glob.glob(os.path.join(rootdir,join,'atlases')+'/*/'+atlasfile);

        print('loading the atlas'),
        atl=nib.load(atlaspath[0]);
        atl_data=atl.get_data();

        if npzfile.count('_plus_'):
            atl_data=np.concatenate((atl_data,atl_data+atl_data.max()),axis=0);

        print('filling the coefficients'),
        coeff_data=np.zeros(atl_data.shape);
        for t in range(len(feat_opt)):
            coeff_data[np.where(atl_data==feat_opt[t])]=coeff_opt[t];

        print('saving the output'),
        outname=os.path.abspath(npzfile).replace('.npz','_slices.nii.gz');
        outnib=nib.Nifti1Image(coeff_data, atl.affine)
        nib.save(outnib, outname);

        outpng=outname.replace('.nii.gz','.png')
        plot_glass_brain(outnib,threshold=0.5,plot_abs=False,output_file=outpng);

        

    
