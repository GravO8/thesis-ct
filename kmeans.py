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
    shape         = array.shape
    array         = array.reshape((-1,1)).astype("float32")
    _, labels, cluster_centers = cv2.kmeans(array, k, None, CRITERIA, ATTEMPTS, cv2.KMEANS_PP_CENTERS)
    cluster_centers = np.uint8(cluster_centers)
    segmented       = cluster_centers[labels.flatten()]
    return segmented.reshape((shape))


def cmeans(array, c = 3, m = 2):
    ERROR    = 1.0
    MAX_ITER = 10
    SEED     = 0
    shape    = array.shape
    array    = array.reshape((-1,1)).T
    cluster_centers, labels, _, _, _, _, _ = skfuzzy.cmeans(array, c, m, ERROR, MAX_ITER, seed = SEED)
    return labels[1,:].reshape((shape))


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


def save(array, name = "sapo"):
    img = nib.Nifti1Image(array, AFFINE)
    nib.save(img, f"{name}.nii")
    print("done")
    
    
def load_ct(patient_id = 1):
    global AFFINE
    ncct_dir = "../../data/gravo/NCCT"
    path     = os.path.join(ncct_dir, f"{patient_id}.nii")
    ncct     = nib.load(path)
    AFFINE   = ncct.affine
    ncct     = ncct.get_fdata()
    return ncct
    
    
def get_mask(array):
    mask = array.copy()
    mask[mask > 0] = 1
    return mask
    
    
def flip(array):
    return np.flip(array, axis = (0,))
    
    
def fix_tilt(array):
    # adapted from https://medium.com/towards-data-science/medical-image-pre-processing-with-python-d07694852606
    array  = array.astype("uint8")
    mask   = get_mask(array)
    angles = {}
    for i in range(array.shape[-1]):
        a = mask[:,:,i]
        contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            continue
        c  = max(contours, key = cv2.contourArea)
        if len(c) < 5:
            continue
        (x,y),(MA,ma),angle = cv2.fitEllipse(c)
        rmajor = max(MA,ma)/2
        if angle > 90:
            angle -= 90
        else:
            angle += 90
        angle = int(angle)
        if angle in angles:
            angles[angle] += 1
        else:
            angles[angle] = 1
        current_max = max(angles, key = angles.get)
        if current_max == angle:
            M = cv2.getRotationMatrix2D((x, y), angle+180, 1)  #transformation matrix
    for i in range(array.shape[-1]):
        array[:,:,i] = cv2.warpAffine(array[:,:,i], M, (a.shape[1], a.shape[0]), cv2.INTER_CUBIC)
    return array


def cut_edges(array):
    mask = get_mask(array)
    mask = (mask == 1) & (flip(mask) == 1)
    array[mask != 1] = 0
    return array


if __name__ == "__main__":    
    ncct = load_ct(1)
    
    ncct      = fix_tilt(ncct)
    ncct      = cut_edges(ncct)
    segmented = cmeans(ncct, c = 2, m = 2)
    segmented = denoise_2d(segmented*100)
    mirrored  = segmented - flip(segmented)
    
    
    save(mirrored)
