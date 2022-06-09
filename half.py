import nibabel as nib
import os
import numpy as np
from kmeans import TiltFix2

def flip(array):
    return np.flip(array, axis = (0,))

ncct_dir = "../../data/gravo/NCCT"
new_dir = "../../data/half"

if not os.path.isdir(new_dir):
    os.system(f"mkdir {new_dir}")
    
    
for file in os.listdir(ncct_dir):
    if "-" not in file and file.endswith(".nii"):
        print(file)
        path = os.path.join(ncct_dir, file)
        ncct = nib.load(path)
        AFFINE = ncct.affine
        ncct = ncct.get_fdata()
        
        tf = TiltFix2(ncct)
        ncct = tf.bayesian_optimization(N = 90)
        
        half1 = ncct.copy()
        half1[47:,:,:] = 0
        half1 = half1[:46,:,:]
        
        half2 = flip(ncct)
        half2[47:,:,:] = 0
        half2 = half2[:46,:,:]
        
        assert half2.shape[0] == half1.shape[0] == 46
        assert (half1[-1,:,:] == half2[-1,:,:]).all()
        assert (half1[-1,:,:] > 0).any()
        
        id = file.split(".")[0]
        nib.save(nib.Nifti1Image(half1, AFFINE), os.path.join(new_dir, f"{id}-L.nii"))
        nib.save(nib.Nifti1Image(half2, AFFINE), os.path.join(new_dir, f"{id}-R.nii"))
        
        
        
