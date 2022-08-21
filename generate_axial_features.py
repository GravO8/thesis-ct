from models.models import get_timm_model
import os, nibabel as nib, torch, tqdm
# import matplotlib.pyplot as plt

TYPE  = "NCCT"
# NORM  = 100 if TYPE == "NCCT" else 200
# DIR   = "../../data/gravo"
DIR   = f"/media/avcstorage/gravo/{TYPE}_normalized" 
NETS  = ["resnet18", "resnet34", "resnet50"]

for net in NETS:
    model = get_timm_model(net, global_pool = "gap", pretrained = True, frozen = True)
    for file in tqdm.tqdm(os.listdir(DIR)):
        if not file.endswith(".nii"):
            continue
        path_out = os.path.join(DIR, f"{TYPE}_{NET}", f"{file.split('.')[0]}.pt")
        if os.path.isfile(path_out):
            continue
        print(file)
        path = os.path.join(DIR, file)
        scan = torch.tensor(nib.load(path).get_fdata())
        scan = scan.unsqueeze(dim = 0).permute(3,0,1,2).float()
        # scan = scan / NORM
        tensor = model(scan)
        torch.save(tensor, path_out)
