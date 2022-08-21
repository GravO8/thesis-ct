import torch, pandas as pd, nibabel as nib, os, numpy as np, matplotlib.pyplot as plt
from tensor_mil import mil_after_attention, TensorEncoder

CT_TYPE     = "CTA"
NET         = "resnet18"
SET         = "test"
DATA_DIR    = "../../../data/gravo/"
WEIGHT_PATH = "CTA/MILNet-after-TensorAxial-resnet18_gap_2D-AttentionPooling/MILNet-after-TensorAxial-resnet18_gap_2D-AttentionPooling-run3/weights.pt"

DATASET     = os.path.join(DATA_DIR, f"dataset_{CT_TYPE}.csv")
TENSORS_DIR = os.path.join(DATA_DIR, f"{CT_TYPE}_{NET}")
CT_DIR      = os.path.join(DATA_DIR, CT_TYPE)
SKIP_SLICES = 5
N           = 5


model = mil_after_attention(TensorEncoder(NET, 512, "gap"))
model.load_state_dict(torch.load(WEIGHT_PATH))
model.eval()
set  = pd.read_csv(DATASET)
set  = test_set[test_set["set"] == SET]

for _, row in set.iterrows():
    patient = row["patient_id"]
    y_true  = row["binary_rankin"]
    tensor  = torch.load(os.path.join(TENSORS_DIR, f"{patient}.pt"))
    scan    = nib.load(os.path.join(CT_DIR, f"{patient}.nii")).get_fdata()
    i_range = range(0,len(tensor),SKIP_SLICES)
    tensor  = tensor[i_range]
    scan    = scan[...,i_range]
    pred, a = model.predict_attention(tensor)
    a       = a.detach().numpy().squeeze()
    i_att   = np.argsort(a)[::-1]
    print("patient:", patient)
    print(" y_true:", y_true)
    print(" y_pred:", pred)
    print()
    _, axs  = plt.subplots(1, N, figsize = (8,5))
    for i in range(N):
        i_slice = i_att[i]
        axs[i].imshow(np.flip(scan[:,:,i_slice].T,0), cmap = "gray")
        axs[i].axis("off")
        axs[i].title.set_text(round(a[i_slice],3))
    plt.tight_layout()
    plt.show()
    
