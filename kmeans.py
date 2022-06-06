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
    if np.isclose(labels[0,0], 0, atol = .01):
        return labels[0,:].reshape((shape))
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
            angle = -1
            M = cv2.getRotationMatrix2D((x, y), angle, 1)  #transformation matrix
    for i in range(array.shape[-1]):
        array[:,:,i] = cv2.warpAffine(array[:,:,i], M, (a.shape[1], a.shape[0]), cv2.INTER_CUBIC)
    return array


def cut_edges(array):
    mask = get_mask(array)
    mask = (mask == 1) & (flip(mask) == 1)
    array[mask != 1] = 0
    return array
    
    
def mirror(array):
    T = 50
    mirrored                = segmented - flip(segmented)
    # mirrored[mirrored > T]  = T
    # mirrored[mirrored < T]  = 0
    # mirrored[mirrored == T] = 1
    return mirrored
    
    
def rotate_coronal(array):
    slice = array[:,0,:]
    x, y  = np.array(slice.shape)/2
    ANGLE = -5.33856201171875
    M = cv2.getRotationMatrix2D((x, y), ANGLE, 1)  #transformation matrix
    for i in range(array.shape[1]):
        array[:,i,:] = cv2.warpAffine(array[:,i,:], M, (slice.shape[1], slice.shape[0]), cv2.INTER_CUBIC)
    return array
    
def rotate_axial(array):
    slice = array[:,:,0]
    x, y  = np.array(slice.shape)/2
    ANGLE = -3
    M = cv2.getRotationMatrix2D((x, y), ANGLE, 1)  #transformation matrix
    for i in range(array.shape[2]):
        array[:,:,i] = cv2.warpAffine(array[:,:,i], M, (slice.shape[1], slice.shape[0]), cv2.INTER_CUBIC)
    return array
    
    
def test(array):
    import matplotlib.pyplot as plt
    import math
    coronal     = array[:,array.shape[1]//2,:].astype("uint8")
    contours, _ = cv2.findContours(coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt         = max(contours, key = cv2.contourArea)
    rect        = cv2.fitEllipse(cnt)
    ((x,y),(MA,ma),angle) = rect
    print(rect)
    gray = cv2.cvtColor(np.float32(coronal), cv2.COLOR_GRAY2RGB)
    # coronal = cv2.ellipse(gray,rect, color=(0,0,255),thickness =1)
    rmajor = max(MA,ma)/2
    # if angle > 90:
    #     angle -= 90
    # else:
    #     angle += 96
    if angle > 90:
        angle -= 180
    print(angle)
    xtop = x + math.cos(math.radians(angle))*rmajor
    ytop = y + math.sin(math.radians(angle))*rmajor
    xbot = x + math.cos(math.radians(angle+180))*rmajor
    ybot = y + math.sin(math.radians(angle+180))*rmajor
    # cv2.line(coronal, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 1)
    
    M = cv2.getRotationMatrix2D((x, y), angle, 1)  #transformation matrix
    coronal = cv2.warpAffine(coronal, M, (coronal.shape[1],coronal.shape[0]), cv2.INTER_CUBIC)
    plt.imshow(np.flip(coronal.T, axis = (0,)), cmap = "gray")
    plt.show()
    
    
def fix_coronal_rotation(array):
    middle_i = array.shape[1]//2
    x_cml, y_cml, angle_cml = [], [], []
    range_ = list(range(0,20,4))
    for j in range_:
        i     = middle_i+j
        slice = array[:,i,:].astype("uint8")
        contours, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt         = max(contours, key = cv2.contourArea)
        rect        = cv2.fitEllipse(cnt)
        (x,y),_,angle = rect
        if angle > 90:
            angle -= 180
        x_cml.append(x)
        y_cml.append(y)
        angle_cml.append(angle)
    x, y, angle = np.array(x_cml), np.array(y_cml), np.array(angle_cml)
    if angle.std() > 3:
        print(x)
        print( angle )
        return array
    print("rotated")
    M = cv2.getRotationMatrix2D((x.mean(), y.mean()), angle.mean(), 1)  #transformation matrix
    for i in range(array.shape[1]):
        array[:,i,:] = cv2.warpAffine(array[:,i,:], M, (slice.shape[1], slice.shape[0]), cv2.INTER_CUBIC)
    return array
    

def get_brain_outline(array):
    cnt = []
    for z in range(array.shape[2]):
        for y in range(array.shape[1]):
            for x in range(array.shape[0]):
                if array[x][y][z] > 0:
                    cnt.append( (x,y,z) )
                    break
            for x in range(array.shape[0]-1,0,-1):
                if array[x][y][z] > 0:
                    cnt.append( (x,y,z) )
                    break
    for z in range(array.shape[2]):
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                if array[x][y][z] > 0:
                    cnt.append( (x,y,z) )
                    break
            for y in range(array.shape[1]-1,0,-1):
                if array[x][y][z] > 0:
                    cnt.append( (x,y,z) )
                    break
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                if array[x][y][z] > 0:
                    cnt.append( (x,y,z) )
                    break
            for z in range(array.shape[2]-1,0,-1):
                if array[x][y][z] > 0:
                    cnt.append( (x,y,z) )
                    break
    return cnt
    
    
def extend_line(p1, p2, distance=10000):
    # copied from https://stackoverflow.com/a/72084363 
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance*np.cos(diff))
    p3_y = int(p1[1] + distance*np.sin(diff))
    p4_x = int(p1[0] - distance*np.cos(diff))
    p4_y = int(p1[1] - distance*np.sin(diff))
    return ((p3_x, p3_y), (p4_x, p4_y))    
    
def test_pca(array):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    coronal     = array[:,array.shape[1]//2,:].astype("uint8")
    contours, _ = cv2.findContours(coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt         = max(contours, key = cv2.contourArea).squeeze()
    cnt = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if array[i][j][k] > 0:
                    cnt.append( (i,j,k) )
    pca = PCA(n_components = 3)
    pca.fit(cnt)
    x, y, z = [round(v) for v in pca.mean_]
    components = (pca.components_*100000).astype(int)
    print(pca.mean_, x, y, z)
    print(pca.components_)
    print(pca.explained_variance_)
    coronal = cv2.cvtColor(np.float32(coronal), cv2.COLOR_GRAY2RGB)
    line_   = extend_line((x, z), tuple(components[1,[0,2]])) 
    coronal = cv2.line(coronal, line_[0], line_[1], (0, 255, 0), thickness=1)
    line_   = extend_line((x, z), tuple(components[2,[0,2]])) 
    coronal = cv2.line(coronal, line_[0], line_[1], (0, 255, 0), thickness=1)
    plt.imshow(np.flip(coronal[:,:,1].T, axis = (0,)), cmap = "gray")
    # plt.imshow(np.flip(coronal.T, axis = (0,)), cmap = "gray")
    plt.show()
    

if __name__ == "__main__":
    # for file in [f for f in os.listdir("../../data/gravo/NCCT") if "-" not in f]:
        # ncct = load_ct(int(file.split(".")[0]))
    ncct = load_ct(131026)
    # 46149
    
    test_pca(ncct)
    # ncct = fix_coronal_rotation(ncct)
    # ncct      = fix_tilt(ncct)
    # ncct      = cut_edges(ncct)
    # segmented = cmeans(ncct, c = 2, m = 2)
    # segmented = denoise_2d(segmented*100)
    # mirrored  = mirror(segmented)
    
    # test(ncct)
    # save(ncct)
    # save(mirrored)
    # save(segmented)
