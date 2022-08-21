import torch, pandas as pd, nibabel as bib
from tensor_mil import mil_after_attention

CT_TYPE     = "CTA"
SKIP_SLICES = 5

model       = mil_after_attention()
test_set    = pd.read_csv(f"../../../data/gravo/dataset_{CT_TYPE}.csv")
test_set    = test_set[test_set["set"] == "test"]

for _, row in test_set.iterrows():
    patient = row["patient_id"]
    
