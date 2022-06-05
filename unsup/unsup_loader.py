import sys, os, torchio
sys.path.append("..")
import numpy as np
import pandas as pd
from utils.ct_loader import BINARY_RANKIN, RANKIN, PATIENT_ID, CTLoader


TRANSFORM = torchio.Compose([torchio.RandomFlip("lr", p = 0.5), 
                            torchio.RandomAffine(scales = 0, translation = 0, degrees = 5, center = "image", p = 1),
                            torchio.RandomElasticDeformation(p = 0.2),
                            torchio.RandomAnisotropy(downsampling = 1.5, p = .2),
                            torchio.RandomNoise(mean = 5, std = 2, p = 0.2),
                            torchio.RandomGamma(p = 1)])


class UnSupCTLoader(CTLoader):
    def __init__(self, labels_filename: str = "dataset.csv", data_dir: str = None,
        binary_rankin: bool = True, batch_size: int = 32):
        self.data_dir       = "" if data_dir is None else data_dir
        self.label_col      = BINARY_RANKIN if binary_rankin else RANKIN
        self.labels         = pd.read_csv( os.path.join(self.data_dir, labels_filename) )
        self.batch_size     = batch_size
        self.augment_train  = False
        np.random.seed(0)

    def load_train(self):
        val_test_ids = self.labels[(self.labels["set"] == "val") | (self.labels["set"] == "test")]["patient_id"].values
        ncct_dir     = os.path.join(self.data_dir, "NCCT")
        train_set    = []
        nccts        = os.listdir(ncct_dir)
        nccts        = [file for file in nccts if ("-" not in file) and file.endswith(".nii")]
        nccts        = [file for file in nccts if int(file.split(".")[0]) not in val_test_ids]
        nccts        = nccts[:-(len(nccts)%self.batch_size)]
        i            = 0
        for file in nccts:
            patient_id = int(file.split(".")[0])
            path        = os.path.join(ncct_dir, f"{patient_id}.nii")
            subject     = torchio.Subject(
                ct          = torchio.ScalarImage(path),
                patient_id  = patient_id,
                target      = np.nan,
                transform   = "original")
            train_set.append(subject)
            i += 1
            if (i % self.batch_size) == 0:
                for j in range(1,self.batch_size+1):
                    augment              = TRANSFORM( train_set[-j] )
                    augment["transform"] = "augment"
                    train_set.append( augment )
        return train_set
