import sys, torch
sys.path.append("..")
from utils.half_ct_loader import HalfCTLoader
from utils.main import set_home
from models.vae import vae_v1
import nibabel as nib


if __name__ == "__main__":
    dir     = set_home(0)
    shape   = (64, 128, 128)
    loader  = HalfCTLoader(data_dir = dir, pad = shape)
    model   = vae_v1(shape = shape, n_start_chans = 8, N = 6)
    
    path    = "../../../runs/systematic/vae/vae_v1/vae_v1-run3/weights.pt"
    model.load_state_dict(torch.load(path, map_location = torch.device("cpu")))
    
    train, _, _ = loader.load_dataset()
    train = torch.utils.data.DataLoader(train, 
            batch_size  = 1, 
            num_workers = 0, 
            pin_memory  = False)
    i = 0
    for batch in train:
        if i < 100: 
            i += 1
            continue
        x      = batch["ct"]["data"].long()
        affine = batch["ct"]["affine"][0]
        model.eval()
        x_line, _, _ = model(x)
        nib.save(nib.Nifti1Image(x.numpy().squeeze(), affine), "1.nii")
        nib.save(nib.Nifti1Image(x_line.detach().numpy().squeeze(), affine), "2.nii")
        break
