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
        binary_rankin: bool = True, augment_train: bool = True, 
        skip_slices: int = 0, reshuffle: bool = False, kfold: bool = False, 
        clip: bool = False):
        self.data_dir       = data_dir
        self.label_col      = BINARY_RANKIN if binary_rankin else RANKIN
        self.augment_train  = augment_train
        if self.data_dir is not None:
            labels_filename         = os.path.join(self.data_dir, labels_filename)
            augmentations_filename  = os.path.join(self.data_dir, augmentations_filename)
        self.labels         = pd.read_csv(labels_filename)
        if reshuffle:
            if kfold:
                self.labels = self.labels.sample(frac = 1).reset_index(drop = True)
            else:
                self.reshuffle()
        self.original_sets  = list(self.labels[SET].values)
        self.augmentations  = pd.read_csv(augmentations_filename)
        self.skip_slices    = skip_slices
        self.kfold          = kfold
        self.clip           = clip
        np.random.seed(0)
        
    def get_folds(self):
        '''
        5 folds and 1 test set
        '''
        self.labels[SET] = self.original_sets # ensures we use the same folds every time the get_folds method is called
        patients1 = self.labels[self.labels[BINARY_RANKIN] == 1]
        patients0 = self.labels[self.labels[BINARY_RANKIN] == 0]
        N1_ids    = list(patients1[PATIENT_ID].values)
        N0_ids    = list(patients0[PATIENT_ID].values)
        N1_fold   = len(patients1) // 6
        N0_fold   = len(patients0) // 6
        N1_train  = N1_fold * 5
        N0_train  = N0_fold * 5
        for i in range(5):
            test_patients   = N1_ids[i*N1_fold:(i+1)*N1_fold] + N0_ids[i*N0_fold:(i+1)*N0_fold]
            train_patients  = N1_ids[:i*N1_fold] + N1_ids[(i+1)*N1_fold:N1_train]
            train_patients += N0_ids[:i*N0_fold] + N0_ids[(i+1)*N0_fold:N0_train]
            new_set = []
            for id in self.labels["patient_id"].values:
                if id in test_patients:
                    new_set.append("test")
                elif id in train_patients:
                    new_set.append("train")
                else:
                    new_set.append(None)
            self.labels[SET] = new_set
            yield self.load_dataset()
        train_patients = N1_ids[:N1_train] + N0_ids[:N0_train]
        self.labels[SET] = ["train" if id in train_patients else "test" for id in self.labels["patient_id"].values]
        yield self.load_dataset()
        
    def reshuffle(self, verbose = False):
        if verbose:
            print("before")
            print(" - train", np.unique(self.labels[self.labels[SET] == "train"][BINARY_RANKIN].values, return_counts = True))
            print(" - test", np.unique(self.labels[self.labels[SET] == "test"][BINARY_RANKIN].values, return_counts = True))
            print(self.labels[self.labels[SET] == "test"][:5])
        patients1   = self.labels[self.labels[BINARY_RANKIN] == 1]
        patients0   = self.labels[self.labels[BINARY_RANKIN] == 0]
        N1_ids      = list(patients1[PATIENT_ID].values)
        N0_ids      = list(patients0[PATIENT_ID].values)
        N1_test     = len(patients1[patients1[SET] == "test"].values)
        N0_test     = len(patients0[patients0[SET] == "test"].values)
        np.random.shuffle(N0_ids)
        np.random.shuffle(N1_ids)
        test_patients = N1_ids[:N1_test] + N0_ids[:N0_test]
        self.labels[SET] = ["test" if id in test_patients else "train" for id in self.labels["patient_id"].values]
        if verbose:
            print("after")
            print(" - train", np.unique(self.labels[self.labels[SET] == "train"][BINARY_RANKIN].values, return_counts = True))
            print(" - test", np.unique(self.labels[self.labels[SET] == "test"][BINARY_RANKIN].values, return_counts = True))
            print(self.labels[self.labels[SET] == "test"][:5])
            
    def load_dataset(self, verbose = True):
        train   = self.load_set("train")
        test    = self.load_set("test")
        if self.augment_train:
            train_augmentations = self.load_train_augmentations()
            train.extend( train_augmentations )
        np.random.shuffle(train)
        np.random.shuffle(test)
        if verbose:
            print("train", len(train), np.unique(self.labels[self.labels[SET] == "train"][BINARY_RANKIN].values, return_counts = True))
            print("test", len(test), np.unique(self.labels[self.labels[SET] == "test"][BINARY_RANKIN].values, return_counts = True))
        return (torchio.SubjectsDataset(train),
                torchio.SubjectsDataset(test))
                
    def load_train_augmentations(self):
        augmentations_set = []
        for _, row in self.labels[self.labels["set"] == "train"].iterrows():
            patient_id = row[PATIENT_ID]
            for augmentation in ("flip", "elastic_deformation", "flip_elastic_deformation"):
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
        if self.clip:
            tensor = tensor[15:70,:]
        if self.skip_slices > 1:
            tensor = tensor[range(0,len(tensor),self.skip_slices),:]
        tensor = tensor.unsqueeze(dim = 0).unsqueeze(dim = 0)
        return torchio.ScalarImage(tensor = tensor)


if __name__ == "__main__":
    # ct_loader = CTLoader2D("A", data_dir = "../../../data/gravo")
    # data_dir = "../../../data/gravo"
    data_dir = "/media/avcstorage/gravo"
    ct_loader = CTLoaderTensors(data_dir = data_dir, reshuffle = True)
    for train, test in ct_loader.get_folds():
        pass
    # ct_loader.load_dataset()
    # 57217
