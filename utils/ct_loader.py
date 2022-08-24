import os, torch, torchio, numpy as np, pandas as pd
from .dataset_splitter import PATIENT_ID, RANKIN, BINARY_RANKIN, AUGMENTATION, SET

CT_TYPE         = "NCCT"
LABELS_FILENAME = f"dataset_{CT_TYPE}.csv"
AUGMENTATIONS   = f"augmentations_{CT_TYPE}.csv"


def add_pad(scan, pad: int):
    scan    = scan.unsqueeze(dim = 0)
    scan    = torch.nn.functional.interpolate(scan, scale_factor = 2, mode = "bilinear")
    scan    = scan.squeeze(dim = 0)
    _, w, h = scan.shape
    zeros   = torch.zeros(1, pad, pad)
    pad_w   = (pad-w)//2
    pad_h   = (pad-h)//2
    zeros[:, pad_w:pad_w+w, pad_h:pad_h+h] = scan
    scan = zeros
    return scan
    

class CTLoader:
    def __init__(self, labels_filename: str = LABELS_FILENAME, 
        augmentations_filename = AUGMENTATIONS, data_dir: str = None,
        binary_rankin: bool = True, augment_train: bool = True, skip_slices: int = 0):
        self.data_dir       = data_dir
        self.label_col      = BINARY_RANKIN if binary_rankin else RANKIN
        self.augment_train  = augment_train
        if self.data_dir is not None:
            labels_filename         = os.path.join(self.data_dir, labels_filename)
            augmentations_filename  = os.path.join(self.data_dir, augmentations_filename)
        self.labels         = pd.read_csv(labels_filename)
        self.augmentations  = pd.read_csv(augmentations_filename)
        self.skip_slices    = skip_slices
        np.random.seed(0)
            
    def load_dataset(self):
        train   = self.load_set("train")
        test    = self.load_set("test")
        if self.augment_train:
            train_augmentations = self.load_train_augmentations()
            train.extend( train_augmentations )
        np.random.shuffle(train)
        np.random.shuffle(test)
        print(len(train), len(test))
        return (torchio.SubjectsDataset(train),
                torchio.SubjectsDataset(test))
                
    def load_train_augmentations(self):
        augmentations_set = []
        for _, row in self.labels[self.labels["set"] == "train"].iterrows():
            patient_id = row[PATIENT_ID]
            for augmentation in self.augmentations[self.augmentations[PATIENT_ID] == patient_id][AUGMENTATION].values:
                ct      = self.get_ct(f"{patient_id}-{augmentation}")
                subject = torchio.Subject(
                    ct          = ct,
                    patient_id  = patient_id,
                    target      = row[self.label_col],
                    transform   = augmentation)
                augmentations_set.append(subject)
        return augmentations_set
    
    def load_set(self, set_name: str):
        assert set_name in ("train", "test")
        set = []
        for _, row in self.labels[self.labels["set"] == set_name].iterrows():
            patient_id  = row[PATIENT_ID]
            ct          = self.get_ct(patient_id)
            subject     = torchio.Subject(
                ct          = ct,
                patient_id  = patient_id,
                target      = row[self.label_col],
                transform   = "original")
            set.append(subject)
        return set
        
    def get_ct(self, patient_id):
        path = os.path.join(self.data_dir, CT_TYPE, f"{patient_id}.nii")
        ct   = torchio.ScalarImage(path)
        # ct.set_data( ct.data[:,:,:,[i for i in range(ct.shape[-1]) if torch.count_nonzero(ct.data[...,i] > 0) > 100]] )
        if self.skip_slices > 0:
            ct.set_data(ct.data[...,range(0,ct.shape[-1],self.skip_slices)])
        return ct
        
        
class CTLoader2D(CTLoader):
    def __init__(self, slice: str, slice_range: int = 2, pad: int = None,
        **kwargs):
        super().__init__(**kwargs)
        assert slice_range >= 0
        self.pad              = pad
        self.slice_range      = slice_range
        self.slice            = slice
        self.reference_scans  = (2243971, 2520986, 2605128, 2505743, 1911947)
        self.reference_slices = {
            "A": [35, 33, 34, 36, 39],
            "B": [50, 47, 48, 52, 51],
            "C": [25, 26, 24, 25, 30]
        }
        assert self.slice in [r for r in self.reference_slices], "CTLoader2D__init__: slice must be in 'A','B' or 'C'."
        self.set_mask()

    def set_mask(self):
        self.mask = None
        for i in range(len(self.reference_scans)):
            patient_id  = self.reference_scans[i]
            axial_slice = self.reference_slices[self.slice][i]
            assert self.labels[self.labels[PATIENT_ID] == patient_id][SET].values[0] == "train", "CTLoader2D.set_mask: all reference scans must be from the train set."
            path        = os.path.join(self.data_dir, CT_TYPE, f"{patient_id}.nii")
            scan        = torchio.ScalarImage(path)[torchio.DATA].float()
            if self.mask is None:
                self.mask  = scan[:,:,:,axial_slice]
            else:
                self.mask += scan[:,:,:,axial_slice]
        self.mask    /= len(self.reference_scans)
        self.negative = self.mask.max() - self.mask
        
    def load_set(self, set_name: str):
        set = super().load_set(set_name)
        set = self.select_slice(set)
        return set
        
    def load_train_augmentations(self):
        train_augmentations = super().load_train_augmentations()
        train_augmentations = self.select_slice(train_augmentations)
        return train_augmentations
        
    def select_slice(self, set, debug = False):
        for subject in set:
            scores = []
            scan   = subject["ct"][torchio.DATA]
            for i in range(scan.shape[-1]):
                ax_slice = scan[:,:,:,i].squeeze() # shape = (B,x,y,z)
                score    = (ax_slice*self.mask).sum() - (ax_slice*self.negative).sum()
                scores.append(score)
            i      = np.argmax(scores)
            scan   = scan[:,:,:,i-self.slice_range:i+self.slice_range+1].mean(axis = 3)
            scan   = scan if self.pad is None else add_pad(scan, self.pad)
            scan   = scan.unsqueeze(dim = -1) # torchio expects 4D tensors
            subject["ct"][torchio.DATA] = scan
            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(scan.squeeze().T.flip(0), cmap = "gray")
                plt.show()
        return set
            

class CTLoaderTensors(CTLoader):
    def __init__(self, encoder: str = "resnet18", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
    
    def get_ct(self, patient_id):
        path   = os.path.join(self.data_dir, f"{CT_TYPE}_normalized/{CT_TYPE}_{self.encoder}", f"{patient_id}.pt")
        tensor = torch.load(path)
        tensor = tensor[15:70,:]
        if self.skip_slices > 0:
            tensor = tensor[range(0,len(tensor),self.skip_slices),:]
        tensor = tensor.unsqueeze(dim = 0).unsqueeze(dim = 0)
        return torchio.ScalarImage(tensor = tensor)

if __name__ == "__main__":
    # ct_loader = CTLoader2D("A", data_dir = "../../../data/gravo")
    ct_loader = CTLoaderTensors(data_dir = "../../../data/gravo")
    ct_loader.load_dataset()
    # 57217
