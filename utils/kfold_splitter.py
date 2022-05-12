import os
import pandas as pd
import numpy as np

DATA_DIR        = "../../../data/gravo"
K_FOLDS         = 5
TEST_SIZE       = 60
DATASET_SIZE    = 465


def get_patients(table_data: pd.DataFrame, nccts: list):
    patients = {"patient_id":[], "rankin":[], "binary_rankin": []}
    for _, row in table_data.iterrows():
        patient_id  = row["idProcessoLocal"]
        rankin      = row["rankin-23"]
        if rankin.isnumeric() and (row["NCCT"] == "OK"):
            assert f"{patient_id}.nii" in nccts, f"get_patients: {patient_id} doesn't have a NCCT scan."
            rankin = int(rankin)
            patients["patient_id"].append(patient_id)
            patients["rankin"].append(rankin)
            patients["binary_rankin"].append(int(rankin > 2))
    return pd.DataFrame(patients)
    
    
def kfold_split(dataset: pd.DataFrame):
    assert len(dataset) == DATASET_SIZE, "kfold_split: unexpected dataset size"
    dataset    = dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
    N0, N1     = np.bincount(dataset["binary_rankin"].values)
    N1_TEST    = int(N1*TEST_SIZE/DATASET_SIZE)
    N0_TEST    = TEST_SIZE - N1_TEST
    N1_VAL     = N1 - N1_TEST
    N0_VAL     = N0 - N0_TEST
    N1_FOLD    = int(N1_VAL/K_FOLDS)
    N0_FOLD    = int(N0_VAL/K_FOLDS)
    sets       = []
    n1_test    = 0
    n0_test    = 0
    n1_val     = 0
    n0_val     = 0
    for _, row in dataset.iterrows():
        if row["binary_rankin"] == 1:
            if n1_test < N1_TEST:
                sets.append("test")
                n1_test += 1
            else:
                sets.append(f"fold{(n1_val//N1_FOLD)+1}")
                n1_val += 1
        else:
            if n0_test < N0_TEST:
                sets.append("test")
                n0_test += 1
            else:
                sets.append(f"fold{(n0_val//N0_FOLD)+1}")
                n0_val += 1
    dataset["set"] = sets
    def assert_dataset():
        assert TEST_SIZE == len(dataset[dataset["set"] == "test"])
        assert N1_TEST == len(dataset[(dataset["set"] == "test") & (dataset["binary_rankin"] == 1)])
        assert N0_TEST == len(dataset[(dataset["set"] == "test") & (dataset["binary_rankin"] == 0)])
        for k in range(1, K_FOLDS+1):
            assert N1_FOLD == len(dataset[(dataset["set"] == f"fold{k}") & (dataset["binary_rankin"] == 1)])
            assert N0_FOLD == len(dataset[(dataset["set"] == f"fold{k}") & (dataset["binary_rankin"] == 0)])
    assert_dataset()
    return dataset
    
    
def prettify_dataset(dataset: pd.DataFrame):
    sort_col = [0 if row["set"] == "test" else int(row["set"].replace("fold", "")) for _, row in dataset.iterrows()]
    dataset["sort"] = sort_col
    dataset = dataset.sort_values(by = ["sort", "rankin"])
    del dataset["sort"]
    return dataset
    

if __name__ == "__main__":
    table_data_dir  = os.path.join(DATA_DIR, "table_data.csv")
    nccts_dir       = os.path.join(DATA_DIR, "NCCT")
    table_data      = pd.read_csv(table_data_dir)
    nccts           = [file for file in os.listdir(nccts_dir) if file.endswith(".nii")]
    
    dataset         = get_patients(table_data, nccts)
    dataset         = kfold_split(dataset)
    dataset         = prettify_dataset(dataset)
    dataset.to_csv("dataset.csv", index = False)
