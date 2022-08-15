from models.models import get_timm_model
import os, nibabel as nib, torch, tqdm
# import matplotlib.pyplot as plt

TYPE  = "CTA"
# DIR   = "../../data/gravo"
DIR   = "/media/avcstorage/gravo" 
NET   = "resnet50"
model = get_timm_model(NET, global_pool = "gap", pretrained = True, frozen = True)

for file in tqdm.tqdm(os.listdir(os.path.join(DIR, TYPE))):
    if not file.endswith(".nii"):
        continue
    path = os.path.join(DIR, TYPE, file)
    scan = torch.tensor(nib.load(path).get_fdata())
    scan = scan.unsqueeze(dim = 0).permute(3,0,1,2).float()
    scan = scan / 200
    tensor = model(scan)
    path = os.path.join(DIR, f"{TYPE}_{NET}", f"{file.split('.')[0]}.pt")
    torch.save(tensor, path)
