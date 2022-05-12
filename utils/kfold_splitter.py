import os
import pandas as pd
import numpy as np

DATA_DIR        = "../../../data/gravo"
K_FOLDS         = 5
TEST_SIZE       = 60
DATASET_SIZE    = 465
PATIENT_ID      = "patient_id"
RANKIN          = "rankin"
BINARY_RANKIN   = "binary_rankin"
AUGMENTS        = ["RandomFlip", "RandomNoise", "RandomElasticDeformation", "RandomAffine"]


def get_patients(table_data: pd.DataFrame, nccts: list):
    patients = {PATIENT_ID:[], RANKIN:[], BINARY_RANKIN: []}
    for _, row in table_data.iterrows():
        patient_id  = row["idProcessoLocal"]
        rankin      = row["rankin-23"]
        if rankin.isnumeric() and (row["NCCT"] == "OK"):
            assert f"{patient_id}.nii" in nccts, f"get_patients: {patient_id} doesn't have a NCCT scan."
            rankin = int(rankin)
            patients[PATIENT_ID].append(patient_id)
            patients[RANKIN].append(rankin)
            patients[BINARY_RANKIN].append(int(rankin > 2))
    return pd.DataFrame(patients)
    
    
def kfold_split(dataset: pd.DataFrame):
    '''
    +-------+-------+-------+-------+-------+------+
    | fold1 | fold2 | fold3 | fold4 | fold5 | test |
    +-------+-------+-------+-------+-------+------+
    is equivalent to
    +-------+-------+-------+-------+-------+------+
    |              development              | test |
    +-------+-------+-------+-------+-------+------+
    '''
    assert len(dataset) == DATASET_SIZE, "kfold_split: unexpected dataset size"
    dataset    = dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
    N0, N1     = np.bincount(dataset[BINARY_RANKIN].values) # N1 = number of examples of class 1
    N1_TEST    = int(N1*TEST_SIZE/DATASET_SIZE)               # N1_TEST = number of examples of class 1 in the test set
    N0_TEST    = TEST_SIZE - N1_TEST
    N1_DEV     = N1 - N1_TEST                                 # N1_DEV = number of examples of class 1 in the development set
    N0_DEV     = N0 - N0_TEST
    N1_FOLD    = int(N1_DEV/K_FOLDS)                          # N1_FOLD = number of examples of class 1 in each fold
    N0_FOLD    = int(N0_DEV/K_FOLDS)
    sets       = []
    n1_test    = 0
    n0_test    = 0
    n1_dev     = 0
    n0_dev     = 0
    for _, row in dataset.iterrows():
        if row[BINARY_RANKIN] == 1:
            if n1_test < N1_TEST:
                sets.append("test")
                n1_test += 1
            else:
                sets.append(f"fold{(n1_dev//N1_FOLD)+1}")
                n1_dev += 1
        else:
            if n0_test < N0_TEST:
                sets.append("test")
                n0_test += 1
            else:
                sets.append(f"fold{(n0_dev//N0_FOLD)+1}")
                n0_dev += 1
    dataset["set"] = sets
    def assert_dataset():
        assert TEST_SIZE == len(dataset[dataset["set"] == "test"])
        assert N1_TEST == len(dataset[(dataset["set"] == "test") & (dataset[BINARY_RANKIN] == 1)])
        assert N0_TEST == len(dataset[(dataset["set"] == "test") & (dataset[BINARY_RANKIN] == 0)])
        for k in range(1, K_FOLDS+1):
            assert N1_FOLD == len(dataset[(dataset["set"] == f"fold{k}") & (dataset[BINARY_RANKIN] == 1)])
            assert N0_FOLD == len(dataset[(dataset["set"] == f"fold{k}") & (dataset[BINARY_RANKIN] == 0)])
    assert_dataset()
    return dataset
    
    
def prettify_dataset(dataset: pd.DataFrame):
    sort_col        = [0 if row["set"] == "test" else int(row["set"].replace("fold", "")) for _, row in dataset.iterrows()]
    dataset["sort"] = sort_col
    dataset         = dataset.sort_values(by = ["sort", RANKIN])
    del dataset["sort"]
    return dataset
    
    
def get_augmentations(dataset: pd.DataFrame):
    augmentations   = {"fold":[], PATIENT_ID:[], "augmentation":[]}
    n_augments      = len(AUGMENTS)
    for k in range(1,K_FOLDS+1):
        fold_set    = dataset[dataset["set"] == f"fold{k}"]
        n_fold      = np.bincount(fold_set[BINARY_RANKIN].values)
        n_augment   = n_fold[1]*n_augments # 1 is the minority class
        for label in (0, 1):
            ids                 = fold_set[fold_set[BINARY_RANKIN] == label][PATIENT_ID].values
            to_add              = n_augment - n_fold[label]
            transforms_to_add   = [to_add // n_augments] * n_augments
            for i in range(to_add % n_augments):
                transforms_to_add[i] += 1
            for i in range(n_augments):
                for id in np.random.choice(ids, size = transforms_to_add[i], replace = False):
                    augmentations["fold"].append(k)
                    augmentations[PATIENT_ID].append(id)
                    augmentations["augmentation"].append(AUGMENTS[i])
    augmentations = pd.DataFrame(augmentations)
    print(augmentations)
    

if __name__ == "__main__":
    table_data_dir  = os.path.join(DATA_DIR, "table_data.csv")
    nccts_dir       = os.path.join(DATA_DIR, "NCCT")
    table_data      = pd.read_csv(table_data_dir)
    nccts           = [file for file in os.listdir(nccts_dir) if file.endswith(".nii")]
    
    dataset         = get_patients(table_data, nccts)
    dataset         = kfold_split(dataset)
    dataset         = prettify_dataset(dataset)
    
    get_augmentations(dataset)
    # dataset.to_csv("dataset.csv", index = False)
