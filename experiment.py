import torch, torchio
import nibabel as nib
from utils.ct_loader import CTLoader


dir = "../../data/gravo"
ct_loader           = CTLoader(data_dir = dir, augment_train = False)
train, val, test    = ct_loader.load_dataset()
train_loader        = torch.utils.data.DataLoader(train, 
                            batch_size  = 2, 
                            num_workers = 0)

transform = torchio.Compose([torchio.RandomFlip("lr", p = 0.5), 
                            torchio.RandomAffine(scales = 0, translation = 0, degrees = 5, center = "image", p = 1),
                            torchio.RandomElasticDeformation(p = 0.2),
                            torchio.RandomAnisotropy(downsampling = 1.5, p = .2),
                            torchio.RandomNoise(mean = 5, std = 2, p = 0.2),
                            torchio.RandomGamma(p = 1)])
                            
def save(data, affine, name):
    data = data.squeeze()
    nib.save(nib.Nifti1Image(data.numpy(), affine), name)
                        
for batch in train_loader:
    scans = batch["ct"]["data"]
    affine = batch["ct"]["affine"][0].squeeze()
    for i in range(scans.shape[0]):
        a = scans[i]
        save(a, affine, "sapo-original.nii")
        for _ in range(5):
            b = transform(a)
            save(b, affine, "sapo.nii")
            input()
    break
