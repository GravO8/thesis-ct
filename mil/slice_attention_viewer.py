import torch, pandas as pd, nibabel as nib, os, numpy as np, matplotlib.pyplot as plt
from tensor_mil import mil_after_attention, TensorEncoder, mil_after_max


def max_important_slices(argmax):
    # slices, counts = np.unique(argmax, return_counts = True)
    # sorted_i = np.argsort(counts)[::-1]
    # return slices[sorted_i], counts
    assert len(argmax) == len(scores)
    slice_scores = {}
    for i in range(len(argmax)):
        slice = argmax[i] # the slice with highest feature i value
        if slice in slice_scores:
            slice_scores[slice] += scores[i]
        else:
            slice_scores[slice] = scores[i]
    slice_scores = dict(sorted(slice_scores.items(), key = lambda item: item[1]))
    a = []
    for i in range(28):
        if i in slice_scores:
            a.append(slice_scores[i])
        else: a.append(0)
    return a


CT_TYPE     = "NCCT"
NET         = "resnet18"
SET         = "test"
DATA_DIR    = "../../../data/gravo/"
WEIGHT_PATH = "MILNet-after-TensorAxial-resnet18_gap_2D-MaxPooling_1Linear_features_1D/MILNet-after-TensorAxial-resnet18_gap_2D-MaxPooling_1Linear_features_1D-run1/weights.pt"
# WEIGHT_PATH = "MILNet-after-TensorAxial-resnet18_gap_2D-AttentionPooling_1Linear_features_1D/MILNet-after-TensorAxial-resnet18_gap_2D-AttentionPooling_1Linear_features_1D-run1/weights.pt"
POOLING    = "Attention" if "Attention" in WEIGHT_PATH else "Max"
assert NET in WEIGHT_PATH

DATASET     = os.path.join(DATA_DIR, f"dataset_{CT_TYPE}.csv")
CT_DIR      = os.path.join(DATA_DIR, f"{CT_TYPE}_normalized")
TENSORS_DIR = os.path.join(CT_DIR, f"{CT_TYPE}_{NET}")
SKIP_SLICES = 2
N           = 5
CLIP        = True

if POOLING == "Attention":
    model = mil_after_attention(TensorEncoder(NET, 512, "gap"))
elif POOLING == "Max":
    model = mil_after_max(TensorEncoder(NET, 512, "gap"))
weights = torch.load(WEIGHT_PATH, map_location = torch.device("cpu"))
model.load_state_dict(weights)
model.eval()
set  = pd.read_csv(DATASET)
set  = set[set["set"] == SET]

if POOLING == "Max":
    encoder      = weights["encoder.feature_extractor.encoder.0.weight"].squeeze().numpy().T
    mlp          = weights["mlp.0.weight"].squeeze().numpy()
    encoder[encoder < 0] = 0
    mlp = np.abs(mlp)
    scores   = encoder.dot(mlp)
    
    

for _, row in set.iterrows():
    patient = row["patient_id"]
    y_true  = row["binary_rankin"]
    tensor  = torch.load(os.path.join(TENSORS_DIR, f"{patient}.pt"))
    scan    = nib.load(os.path.join(CT_DIR, f"{patient}.nii")).get_fdata()
    if CLIP:
        tensor = tensor[15:70,:]
        scan   = scan[:,:,15:70]
    i_range = range(0,len(tensor),SKIP_SLICES)
    tensor  = tensor[i_range]
    scan    = scan[...,i_range]
    pred, a = model.predict_attention(tensor)
    a       = a.detach().numpy().squeeze()
    if POOLING == "Max":
        a = max_important_slices(a)
    # if POOLING == "Attetion":
    i_att = np.argsort(a)[::-1]
    # elif POOLING == "Max":
    #     i_att, a = max_important_slices(a)
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
    
