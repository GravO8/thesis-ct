import nibabel as nib, elasticdeform, numpy as np, matplotlib.pyplot as plt, os, pandas as pd
from skimage import exposure

'''
Elastic Deformations for Data Augmentation in Breast Cancer Mass Detection 
http://www.inescporto.pt/~jsc/publications/conferences/2018EMecaBHI.pdf
'''


CT_TYPE  = "NCCT"
DATA_DIR = "../../data/gravo/"
# DATA_DIR = "/media/avcstorage/gravo"
CT_DIR   = os.path.join(DATA_DIR, CT_TYPE)
CT_OUT   = os.path.join(DATA_DIR, f"{CT_TYPE}_normalized")
DATASET  = os.path.join(DATA_DIR, f"dataset_{CT_TYPE}.csv")

def normalize_01(data):
    data = data-data.min()
    data = data/data.max()
    return data
    
def load(patient_id: int, dir = CT_DIR, augmentation = None):
    filename = f"{patient_id}.nii" if augmentation is None else f"{patient_id}-{augmentation}.nii"
    a = nib.load(os.path.join(dir, filename))
    return a.get_fdata(), a.affine
    
# def load2(patient_id: int, dir = CT_DIR, augmentation = None):
#     filename = f"{patient_id}.nii" if augmentation is None else f"{patient_id}-{augmentation}.nii"
#     a = ants.image_read(os.path.join(dir, filename))
#     return a.numpy()
    
def save(data, affine, patient_id: int, augmentation = None):
    filename = f"{patient_id}.nii" if augmentation is None else f"{patient_id}-{augmentation}.nii"
    nib.save(nib.Nifti1Image(data, affine), os.path.join(CT_OUT, filename))
    
def deform(data, equalized_reference):
    deformed = elasticdeform.deform_random_grid(data, sigma = 0.8, points = 7)
    deformed = normalize_01(deformed)
    deformed = exposure.match_histograms(deformed, equalized_reference)
    return deformed
    
def clahe(data, clip_limit = .009):
    data      = normalize_01(data)
    equalized = exposure.equalize_adapthist(data, clip_limit = clip_limit) # .005
    return equalized
    
def histogram_equalization(data):
    data      = normalize_01(data)
    equalized = exposure.equalize_hist(data)
    return equalized
    
def contrast_strect(data):
    data      = normalize_01(data)
    p2, p98   = np.percentile(data, (2, 98))
    equalized = exposure.rescale_intensity(data, in_range=(p2, p98))
    return equalized
    
def flip(data):
    return np.flip(data, 0)
    
def normalize_HU(data, min_, max_):
    data[data <= min_] = min_
    data[data >= max_] = max_
    return data
    
def augment(patient_id: int, set: str):
    assert set in ("train", "test")
    data, affine = load(patient_id)
    equalized    = clahe(data)
    save(equalized, affine, patient_id)
    if set == "train":
        return
    deformed     = deform(data, equalized)
    save(deformed, affine, patient_id, "elastic_deformation")
    flipped      = flip(data)
    deformed     = deform(flipped, equalized)
    save(deformed, affine, patient_id, "flip_elastic_deformation")
    flipped      = clahe(flipped)
    save(flipped, affine, patient_id, "flip")
    


def normalization_importance():
    N   = 36
    ct1 = 209855 # 
    ct2 = 210117
    # augment(ct1, "test")
    # augment(ct2, "test")
    cts    = [load(ct1)[0], load(ct1, dir = CT_OUT)[0], load(ct2)[0], load(ct2, dir = CT_OUT)[0]]
    cts[0] = normalize_HU(cts[0], 0, 100)
    cts[2] = normalize_HU(cts[2], 0, 79.707)
    fig, axs = plt.subplots(2,4)
    for i in range(len(cts)):
        axs[0,i].imshow(np.flip(cts[i][:,:,N].T,0), cmap = "gray")
        axs[0,i].axis("off")
        axs[1,i].hist(cts[i].ravel(), bins = 100)
        axs[1,i].set_ylim(0,20000)
        axs[1,i].axis("off")
    plt.tight_layout()
    plt.show()

def histogram_vs_clahe():
    N   = 36
    ct  = 1
    # augment(1, "test")
    cts = [load(ct)[0], load(ct, dir = CT_OUT)[0]]
    cts.append(histogram_equalization(cts[0]))
    names = ["(a) Original", "(b) CLAHE", "(c) Histogram\nNormalization"]
    # cts.append(contrast_strect(cts[0]))
    fig, axs = plt.subplots(2, len(cts), gridspec_kw = {"height_ratios": [4, 1]}, figsize = (8,5))
    for i in range(len(cts)):
        axs[0,i].title.set_text(names[i])
        axs[0,i].imshow(np.flip(cts[i][:,:,N].T,0), cmap = "gray")
        axs[0,i].axis("off")
        axs[1,i].hist(cts[i].ravel(), bins = 100)
        axs[1,i].set_ylim(0,20000)
        axs[1,i].axis("off")
    plt.tight_layout()
    plt.show()
    
def elastic_deformation_changes_color():
    import torchio
    N   = 36
    ct  = 1
    # augment(1, "test")
    cts = [load(ct, dir = CT_OUT)[0]]
    el  = elasticdeform.deform_random_grid(cts[0], sigma = .8, points = 7) 
    cts.append(exposure.match_histograms(normalize_01(el),cts[0]))
    cts.append(el)
    cts.append(torchio.RandomElasticDeformation()(cts[0].reshape((1,)+cts[0].shape)).squeeze())
    # cts.append(clahe(cts[2]))
    names = ["(a) CLAHE", "(b) Elastic Deformation\n& Histogram Matching", "(c) Elastic\nDeformation", "(d) TorchIO Elastic\nDeformation"]
    fig, axs = plt.subplots(2, len(cts), gridspec_kw = {"height_ratios": [4, 1]}, figsize = (8,5))
    for i in range(len(cts)):
        axs[0,i].title.set_text(names[i])
        axs[0,i].imshow(np.flip(cts[i][:,:,N].T,0), cmap = "gray")
        axs[0,i].axis("off")
        axs[1,i].hist(cts[i].ravel(), bins = 100)
        axs[1,i].set_ylim(0,20000)
        axs[1,i].axis("off")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # elastic_deformation_changes_color()
    histogram_vs_clahe()
    # dataset = pd.read_csv(DATASET)
    # for _, row in dataset.iterrows():
    #     set = row["set"]
    #     patient_id = row["patient_id"]
    #     augment(patient_id, set)
