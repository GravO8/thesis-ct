import nibabel as nib
import os
import numpy as np
import cv2
import skfuzzy
import torch


def kmeans(array, k = 3):
    CRITERIA_TYPE = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    MAX_ITER      = 10
    EPSILON       = 1.0
    CRITERIA      = (CRITERIA_TYPE, MAX_ITER, EPSILON)
    ATTEMPTS      = 10
    _, labels, cluster_centers = cv2.kmeans(array.astype("float32"), k, None, CRITERIA, ATTEMPTS, cv2.KMEANS_PP_CENTERS)
    cluster_centers = np.uint8(cluster_centers)
    segmented       = cluster_centers[labels.flatten()]
    return segmented


def cmeans(array, c = 3, m = 2):
    ERROR    = 1.0
    MAX_ITER = 10
    SEED     = 0
    cluster_centers, labels, _, _, _, _, _ = skfuzzy.cmeans(array.T, c, m, ERROR, MAX_ITER, seed = SEED)
    return labels[1,:]


def avg_pool(array):
    avg_pool = torch.nn.AvgPool3d(3, stride = 1, padding = 1)
    array    = torch.Tensor(array).unsqueeze(dim = 0)
    array    = avg_pool(array).squeeze().numpy()
    return array


def avg_pool_2d(array):
    avg_pool = torch.nn.AvgPool2d(3, stride = 1, padding = 1)
    array    = torch.Tensor(array).unsqueeze(dim = 0)
    for i in range(array.shape[-1]):
        array[:,:,:,i] = avg_pool(array[:,:,:,i])
    return avg_pool(array).squeeze().numpy()
    
    
def denoise_2d(array):
    for i in range(array.shape[-1]):
        array[:,:,i] = cv2.fastNlMeansDenoising(array[:,:,i].astype("uint8"), None, 10)
    return array

if __name__ == "__main__":    
    ncct_dir  = "../../data/gravo/NCCT"
    path      = os.path.join(ncct_dir, "1.nii")

    ncct      = nib.load(path)
    ncct2D    = ncct.get_fdata().reshape((-1,1))

    segmented = cmeans(ncct2D, c = 2, m = 2)
    segmented = segmented.reshape((ncct.shape))
    segmented = denoise_2d(segmented*100)
    # cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    # segmented = cv2.filter3D(segmented, -1, kernel)
    # segmented = kmeans( (segmented*100).astype(int), k = 2 )

    segmented = nib.Nifti1Image(segmented, ncct.affine)
    nib.save(segmented, "sapo.nii")
    print("done")
