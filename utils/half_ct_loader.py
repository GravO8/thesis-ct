import os, torchio
import pandas as pd
import numpy as np


class HalfCTLoader:
    def __init__(self, labels_filename: str = "dataset.csv", 
    data_dir: str = None, augment_train: bool = False):
        self.data_dir      = "" if data_dir is None else data_dir
        self.augment_train = augment_train
        self.labels        = pd.read_csv(os.path.join(self.data_dir, labels_filename))
        np.random.seed(0)
        
    def create_subject(self, patient_id: int, side: str, transform: str = "original"):
        assert side in ("L", "R")
        path    = os.path.join(self.data_dir, "half", f"{patient_id}-{side}.nii")
        subject = torchio.Subject(
            ct          = torchio.ScalarImage(path),
            patient_id  = patient_id,
            transform   = transform)
        return subject
        
    def load_set(self, set_name: str):
        if set_name == "train":
            return self.load_train_set()
        assert set_name in ("val", "test")
        set = []
        for _, row in self.labels[self.labels["set"] == set_name].iterrows():
            patient_id = row[PATIENT_ID]
            set.append( self.create_subject(patient_id, "L") )
            set.append( self.create_subject(patient_id, "R") )
        return set
        
    def load_train_set(self):
        val_test_ids = self.labels[(self.labels["set"] == "val") | (self.labels["set"] == "test")]["patient_id"].values
        nccts        = os.listdir( os.path.join(self.data_dir, "half") )
        nccts        = [f for f in nccts if f.endswith("-R.nii")]
        nccts        = [int(f.split("."[0])) for f in nccts]
        train_set    = []
        for ncct in nccts:
            if ncct not in val_test_ids:
                train_set.append( self.create_subject(patient_id, "L") )
                train_set.append( self.create_subject(patient_id, "R") )
        return train_set
        
    def load_dataset(self):
        train   = self.load_set("train")
        val     = self.load_set("val")
        test    = self.load_set("test")
        if self.augment_train:
            assert False, "2 lazy fo dat"
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
        print(len(train), len(val), len(test))
        return (torchio.SubjectsDataset(train),
                torchio.SubjectsDataset(val),
                torchio.SubjectsDataset(test))
