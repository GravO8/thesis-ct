import os, torchio
import numpy as np
import pandas as pd
from ct_loader import BINARY_RANKIN, RANKIN


class UnSupCTLoader:
    def __init__(self, labels_filename: str = "dataset.csv", data_dir: str = None,
        binary_rankin: bool = True):
        self.data_dir   = "" if data_dir is None else self.data_dir
        self.label_col  = BINARY_RANKIN if binary_rankin else RANKIN
        self.labels     = pd.read_csv( os.path.join(self.data_dir, labels_filename) )
        np.random.seed(0)
        
    def load_dataset(self):
        train   = self.load_train("train")
        val     = self.load_not_train("val")
        test    = self.load_not_train("test")
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
        print(len(train), len(val), len(test))
        return (torchio.SubjectsDataset(train),
                torchio.SubjectsDataset(val),
                torchio.SubjectsDataset(test))

    def load_train(self):
        val_test_ids = self.labels[(self.labels["set"] == "val") | (self.labels["set"] == "test")]["patient_id"].values
        ncct_dir     = os.path.join(self.data_dir, "NCCT")
        train_set    = []
        for file in os.listdir(ncct_dir):
            if ("-" not in file) and file.endswith(".nii"):
                id = int(file.split(".")[0])
                if id not in val_test_ids:
                    path        = os.path.join(ncct_dir, f"{patient_id}.nii")
                    subject     = torchio.Subject(
                    ct          = torchio.ScalarImage(path),
                    patient_id  = id,
                    target      = None,
                    transform   = "original")
                    train_set.append(subject)
        return train_set

    def load_not_train(self, set_name: str):
        assert set_name in ("val", "test")
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