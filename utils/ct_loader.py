import os, torchio
import numpy as np
import pandas as pd
from .kfold_splitter import PATIENT_ID, RANKIN, BINARY_RANKIN, AUGMENTATION


class CTLoader:
    def __init__(self, labels_filename: str = "dataset.csv", 
        augmentations_filename = "augmentations.csv", data_dir: str = None,
        binary_rankin: bool = True, augment_train: bool = True):
        self.data_dir       = data_dir
        self.label_col      = BINARY_RANKIN if binary_rankin else RANKIN
        self.augment_train  = augment_train
        if self.data_dir is not None:
            labels_filename         = os.path.join(self.data_dir, labels_filename)
            augmentations_filename  = os.path.join(self.data_dir, augmentations_filename)
        self.labels         = pd.read_csv(labels_filename)
        self.augmentations  = pd.read_csv(augmentations_filename)
            
    def load_dataset(self):
        train   = self.load_set("train")
        val     = self.load_set("val")
        test    = self.load_set("test")
        if self.augment_train:
            train_augmentations = self.load_train_augmentations()
            train.extend( train_augmentations )
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
        # print(len(train), len(val), len(test))
        return (torchio.SubjectsDataset(train),
                torchio.SubjectsDataset(val),
                torchio.SubjectsDataset(test))
                
    def load_train_augmentations(self):
        augmentations_set = []
        for _, row in self.labels[self.labels["set"] == "train"].iterrows():
            patient_id = row[PATIENT_ID]
            for augmentation in self.augmentations[self.augmentations[PATIENT_ID] == patient_id][AUGMENTATION].values:
                path    = os.path.join(self.data_dir, "NCCT", f"{patient_id}-{augmentation}.nii")
                subject = torchio.Subject(
                    ct          = torchio.ScalarImage(path),
                    patient_id  = patient_id,
                    target      = row[self.label_col],
                    transform   = augmentation)
                augmentations_set.append(subject)
        return augmentations_set
    
    def load_set(self, set_name):
        assert set_name in ("train", "val", "test")
        set = []
        for _, row in self.labels[self.labels["set"] == set_name].iterrows():
            patient_id  = row[PATIENT_ID]
            path        = os.path.join(self.data_dir, "NCCT", f"{patient_id}.nii")
            subject     = torchio.Subject(
                ct          = torchio.ScalarImage(path),
                patient_id  = patient_id,
                target      = row[self.label_col],
                transform   = "original")
            set.append(subject)
        return set

    
if __name__ == "__main__":
    ct_loader = CTLoader(data_dir = "../../../data/gravo")
    ct_loader.load_dataset()
