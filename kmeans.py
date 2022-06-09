import nibabel as nib
import os
import numpy as np
import cv2
import skfuzzy
import torch
from scipy import ndimage
from bayes_opt import BayesianOptimization
from abc import ABC


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
    mirrored                = array - flip(array)
    # mirrored[mirrored > T]  = T
    # mirrored[mirrored < T]  = 0
    # mirrored[mirrored == T] = 1
    return mirrored
    
    
def rotate_coronal(input_, angle = -5):
    array = input_.copy()
    slice = array[:,0,:]
    x, y  = np.array(slice.shape)/2
    M = cv2.getRotationMatrix2D((x, y), angle, 1)  #transformation matrix
    for i in range(array.shape[1]):
        array[:,i,:] = cv2.warpAffine(array[:,i,:], M, (slice.shape[1], slice.shape[0]), cv2.INTER_CUBIC)
    return array


def rotate_axial(input_, angle = -3):
    array = input_.copy()
    slice = array[:,:,0]
    x, y  = np.array(slice.shape)/2
    M = cv2.getRotationMatrix2D((x, y), angle, 1)  #transformation matrix
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
    
    
def get_random_point(center: tuple, window = 10):
    pt = []
    for i in range(len(center)):
        dim_center = center[i]
        pt.append( np.random.randint(dim_center-window, dim_center+window) )
    return np.array(pt)
    
    
def get_normal_vector(array, multiplier = 1):
    x = np.random.uniform(.8,1)*multiplier
    y = np.random.uniform(0,.2)*multiplier
    z = np.random.uniform(0,.2)*multiplier
    return np.array([x,y,z])
    
    
def get_bounding_box(array):
    dim_extreme = []
    for i in range(len(array.shape)):
        for j in range(array.shape[i]):
            index = tuple([[j] if c==i else slice(None) for c in range(len(array.shape))])
            if (array[index] > 0).any():
                dim_extreme.append( [j] )
                break
        for j in range(array.shape[i]-1,0,-1):
            index = tuple([[j] if c==i else slice(None) for c in range(len(array.shape))])
            if (array[index] > 0).any():
                dim_extreme[-1].append(j)
                break
    return dim_extreme
    
    
def get_translation(n, d, axis = np.array([1,0,0])):
    '''
    compute the intersection of the plane with the x axis (the axis perpendicular to yOz)
    a.x + b.y + c.z + d = 0    ^    (y = 0 ^ z = 0)
    a.x + d = 0
    x = -d/a
    '''
    e = n[np.argwhere(axis == 1)]
    return (axis*(-d/e)).T
    

def get_d(n, pt):
    '''
    pt = (x0, y0, z0);     n = (a,b,c)
    a(x - x0) + b(y - y0) + c(z - z0) = 0 
    a.x -a.x0 + b.y -b.y0 + c.z -c.z0 = 0
    a.x + b.y + c.z + (-a.x0 -b.y0 -c.z0) = 0
    d = -(a.x0 + b.y0 + c.z0)
    d = - n.pt
    '''
    return - n.dot(pt)
    
    
def angle_between(v1, v2):
    # copied from https://stackoverflow.com/a/13849249
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    

def unit_vector(vector):
    return vector / np.linalg.norm(vector)
    
    
def get_rotation_matrix(n, d, axis = np.array([1,0,0])):
    '''
    returns the rotation matrix that moves the plane defined by the normal vector 
    'n' and the bias point 'd' into the orthonormed plane perpendicular to the 
    'axis' vector
    follows the procedure described here https://math.stackexchange.com/a/1167779
    '''
    t     = get_translation(n, d, axis)
    theta = angle_between(n, axis)
    u     = unit_vector(np.cross(axis, n)) # rotation axis
    # assert u[np.argwhere(axis == 1)] == 0
    cos = np.cos(theta)
    sin = np.sin(theta)
    ux, uy, uz = u
    return np.array(
        [[cos+ux*ux*(1-cos),    ux*uy*(1-cos)-uz*sin, ux*uz*(1-cos)+uy*sin],
         [uy*ux*(1-cos)+uz*sin, cos+uy*uy*(1-cos),    uy*uz*(1-cos)-ux*sin],
         [uz*ux*(1-cos)-uy*sin, uz*uy*(1-cos)*ux*sin, cos+uz*uz*(1-cos)]]), t

     
def mirror_coords(n, d, coords):
    '''
    mirrors the list of points 'coords' along the plane defined by the normal vector
    'n' and the bias point 'd'
    '''
    m, t   = get_rotation_matrix(n, d)
    # m_inv  = np.linalg.inv(m)
    m_inv  = m.T
    mirror = np.array([[-1,0,0],[0,1,0],[0,0,1]])
    return t + m.dot(mirror.dot(m_inv.dot(coords - t)))
        
def save_hemispheres(ncct):
    ncct[ncct > 0] = 1
    msp = ncct.shape[0]//2
    hemisphere1, hemisphere2 = ncct.copy(), ncct.copy()
    hemisphere1[:msp,:,:] = 0
    hemisphere2[msp+1:,:,:] = 0
    hemisphere1 = flip(hemisphere1)
    save(hemisphere1, "hemisphere1")
    save(hemisphere2, "hemisphere2")
        
class TiltFixer(ABC):
    def compare_hemispheres(self, scan = None):
        scan     = self.array if scan is None else scan
        mirrored = flip(scan)
        return self.match(scan, mirrored)
    def match(self, array1, array2):
        '''
        1 - |a - b|
        1 when a == b
        0 when a and b are completely different (0 and 1, for example)
        '''
        mask1 = array1 > 0
        mask2 = array2 > 0
        intersection = mask1 & mask2
        # return intersection.sum()
        # return intersection.sum() - (mask1 & (mask2 == False)).sum() - ((mask1 == False) & mask2).sum()
        return (1 - np.abs(array1[intersection] - array2[intersection])).sum()
        
class MonteCarloTiltFix(TiltFixer):
    def __init__(self, array):
        self.array  = array/array.max()
        self.init_coord_system()
        self.center_scan()
    def center_scan(self):
        '''
        "The center of mass of a body with an axis of symmetry and constant density
        must lie on this axis." - from https://en.wikipedia.org/wiki/Center_of_mass
        This method positions the center of mass of the brain in the center of the
        array
        '''
        self.baseline = self.compare_hemispheres()
        center_of_mass  = ndimage.center_of_mass( (self.array > 0).astype(int) )
        array_center    = []
        for i in range(3):
            array_center.append(self.array.shape[i]//2)
            if self.array.shape[i] % 2 != 0:
                array_center[-1] += 1
        array_center = np.array(array_center)
        translation  = (array_center - np.round(center_of_mass)).astype(int)
        self.array   = np.roll(self.array, translation, axis = (0,1,2))
        self.center  = ndimage.center_of_mass( (self.array > 0).astype(int) )
    def init_coord_system(self):
        '''
        lists of coords of the points in self.array
        used to select the points sliced by the planes and compute their mirrored version
        '''
        self.x, self.y, self.z = [], [], []
        for i in range(self.array.shape[2]):
            for j in range(self.array.shape[1]):
                for k in range(self.array.shape[0]):
                    self.x.append(i)
                    self.y.append(j)
                    self.z.append(k)
        self.x, self.y, self.z = np.array(self.x), np.array(self.y), np.array(self.z)
    def try_plane(self, n, debug = False):
        rotated = self.rotate_brain(n)
        return self.compare_hemispheres(rotated)
    def clip_out_of_bounds(self, coords, other = None):
        '''
        clips the coords_mirrored that are outside the coordinate space defined in self.init_coord_system
        '''
        coords = coords.T
        for i in range(3):
            sel = (coords[:,i] < self.array.shape[i]) & (coords[:,i] > 0)
            if other is not None:
                other = other[sel]
            coords = coords[sel]
        coords = coords.T
        if other is None:
            return coords
        return coords, other
    def find_best_plane(self, N = 1000):
        n           = np.array([1,.0001,0.0001])
        current_max = self.try_plane(n, debug = True)
        best_plane  = None
        for i in range(N):
            n = get_normal_vector(self.array)
            intersection = self.try_plane(n)
            if intersection > current_max:
                intersection = current_max
                best_plane   = (n, d)
    def bayesian_optimization(self, N = 50):
        black_box_function = lambda a,b,c: self.try_plane( np.array([a,b,c]) )
        pbounds            = {"a": (.55, 1), "b": (-.2, .2), "c": (-.2, .2)}
        optimizer = BayesianOptimization(
            f            = black_box_function,
            pbounds      = pbounds,
            random_state = 1)
        optimizer.maximize(init_points = 10, n_iter = N)
        norm_v   = self.try_plane( np.array([1,.0001,.0001]) )
        default  = max(self.baseline, norm_v)
        print("baseline", self.baseline)
        print("default norm vector", norm_v)
        print(optimizer.max)
        print(optimizer.max["target"] > default, optimizer.max["target"]-default)
        best_n = np.array([optimizer.max["params"]["a"],optimizer.max["params"]["b"],optimizer.max["params"]["c"]])
        self.try_plane( best_n, debug = False)
        rotated = self.rotate_brain(best_n)
        return rotated
    def grid_search(self, N = 10):
        a_range = np.linspace(.55, 1, N)
        b_range = np.linspace(-.2, .2, N)
        c_range = np.linspace(-.2, .2, N)
        best = 0
        for a in a_range:
            for b in b_range:
                for c in c_range:
                    score = self.try_plane( np.array([a,b,c]) )
                    print(f"{score}\t{round(a,2)}\t{round(b,2)}\t{round(c,2)}")
                    if score > best:
                        best = score
                        best_n = np.array([a,b,c])
        print("default", self.compare_hemispheres())
        print(best)
        return self.rotate_brain(best_n)
    def rotate_brain(self, n):
        n                         = unit_vector(n)
        d                         = get_d(n, self.center)
        brain                     = self.array > 0
        values                    = self.array[brain]
        brain                     = brain.ravel()
        x_brain, y_brain, z_brain = self.x[brain], self.y[brain], self.z[brain]
        brain_coords              = np.stack([x_brain, y_brain, z_brain], axis = 0)
        m, t                      = get_rotation_matrix(n, d)
        m                         = m.T
        coords_rotated            = t + m.dot(brain_coords - t)
        coords_rotated            = coords_rotated.astype(int)
        coords_rotated, values    = self.clip_out_of_bounds(coords_rotated, values)
        rotated                   = np.zeros(self.array.shape)
        rotated[tuple(coords_rotated)]  = values
        return rotated
        
class TiltFix2(TiltFixer):
    def __init__(self, array):
        self.array = array
        self.baseline = self.compare_hemispheres()
    def try_angles(self, axial_angle, coronal_angle, x):
        rotated = self.rotate_brain(axial_angle, coronal_angle)
        rotated = np.roll(rotated, np.round(x).astype(int), axis = (0,))
        return self.compare_hemispheres(rotated)
    def rotate_brain(self, axial_angle, coronal_angle):
        return rotate_axial(rotate_coronal(self.array, coronal_angle), axial_angle)
    def bayesian_optimization(self, N = 50, verbose = False):
        black_box_function = lambda a1,a2,x: self.try_angles(a1, a2,x)
        pbounds            = {"a1": (-10, 10), "a2": (-10, 10), "x": (-3,3)}
        optimizer = BayesianOptimization(
            f            = black_box_function,
            pbounds      = pbounds,
            random_state = 1,
            verbose      = verbose)
        optimizer.maximize(init_points = 10, n_iter = N)
        if verbose:
            no_angle = self.try_angles(0,0,0)
            print("baseline", self.baseline)
            print("no_angle", no_angle)
            print(optimizer.max)
            print(optimizer.max["target"] > default, optimizer.max["target"]-default)
        if optimizer.max["target"] < self.baseline:
            return self.array
        best_axial, best_coronal, x = optimizer.max["params"]["a1"], optimizer.max["params"]["a2"], optimizer.max["params"]["x"]
        rotated = self.rotate_brain(best_axial, best_coronal)
        rotated = np.roll(rotated, np.round(x).astype(int), axis = (0,))
        return rotated
        

if __name__ == "__main__":
    # for file in [f for f in os.listdir("../../data/gravo/NCCT") if "-" not in f]:
        # ncct = load_ct(int(file.split(".")[0]))
    ncct = load_ct(46149)
    # 120713
    # 131026
    # 46149, 206178 - bastante torto
    
    mc = TiltFix2(ncct)
    ncct = mc.bayesian_optimization(N = 90)
    save(ncct, "rotated")
    # exit(0)
    
    # ncct = fix_coronal_rotation(ncct)
    # ncct      = fix_tilt(ncct)
    ncct      = cut_edges(ncct)
    # mirrored  = mirror(ncct)
    # segmented = kmeans(mirrored)
    segmented = cmeans(ncct, c = 2, m = 2)
    segmented = denoise_2d(segmented*100)
    mirrored  = (mirror(segmented)+100)/100
    # 
    # test(ncct)
    # ncct = rotate_axial(rotate_coronal(ncct))
    # ncct = rotate_axial(ncct, 1)
    # ncct = rotate_coronal(ncct, -2)
    # ncct = np.roll(ncct, 1, axis = (0,))
    # save(ncct)
    save(mirrored)
    # save(segmented)
