import sys, os, torchio
sys.path.append("..")
import numpy as np
import pandas as pd
from utils.ct_loader import BINARY_RANKIN, RANKIN, PATIENT_ID, CTLoader


class UnSupCTLoader(CTLoader):
    def __init__(self, labels_filename: str = "dataset.csv", data_dir: str = None,
        binary_rankin: bool = True):
        self.data_dir   = "" if data_dir is None else data_dir
        self.label_col  = BINARY_RANKIN if binary_rankin else RANKIN
        self.labels     = pd.read_csv( os.path.join(self.data_dir, labels_filename) )
        np.random.seed(0)
        
    def load_dataset(self):
        train         = self.load_train()
        val           = self.load_labeled_set("val")
        test          = self.load_labeled_set("test")
        labeled_train = self.load_labeled_set("train")
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
        # print(len(train), len(val), len(test))
        return (torchio.SubjectsDataset(labeled_train),
                torchio.SubjectsDataset(train),
                torchio.SubjectsDataset(val),
                torchio.SubjectsDataset(test))

    def load_train(self):
        val_test_ids = self.labels[(self.labels["set"] == "val") | (self.labels["set"] == "test")]["patient_id"].values
        ncct_dir     = os.path.join(self.data_dir, "NCCT")
        train_set    = []
        for file in os.listdir(ncct_dir):
            if ("-" not in file) and file.endswith(".nii"):
                patient_id = int(file.split(".")[0])
                if patient_id not in val_test_ids:
                    path        = os.path.join(ncct_dir, f"{patient_id}.nii")
                    subject     = torchio.Subject(
                    ct          = torchio.ScalarImage(path),
                    patient_id  = patient_id,
                    target      = np.nan,
                    transform   = "original")
                    train_set.append(subject)
        return train_set

    def load_labeled_set(self, set_name: str):
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
