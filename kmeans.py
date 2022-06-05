import nibabel as nib
import os
import numpy as np
import cv2

def normalize(segmented):
    i = 0
    for val in np.unique(segmented):
        segmented[segmented == val] = i
        i += 1


ncct_dir  = "../../data/gravo/NCCT"
path      = os.path.join(ncct_dir, "1.nii")

ncct      = nib.load(path)
ncct2D    = ncct.get_fdata().reshape((-1,1)).astype("float32")


K             = 3
CRITERIA_TYPE = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
MAX_ITER      = 10
EPSILON       = 1.0
CRITERIA      = (CRITERIA_TYPE, MAX_ITER, EPSILON)
ATTEMPTS      = 10
_, labels, cluster_centers = cv2.kmeans(ncct2D, K, None, CRITERIA, ATTEMPTS, cv2.KMEANS_PP_CENTERS)

cluster_centers = np.uint8(cluster_centers)
segmented       = cluster_centers[labels.flatten()]
segmented       = segmented.reshape((ncct.shape))
segmented       = nib.Nifti1Image(segmented, ncct.affine)
nib.save(segmented, "sapo.nii")
