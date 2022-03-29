import nibabel as nib
import numpy as np
import torch

class BrainRegions:
    def __init__(self, thickness: int = 1):
        assert thickness in (1, 2), f"BrainRegions.__init__: invalid thickness. Valid values: 1 or 2."
        mni_atlas           = nib.load(f"/usr/local/fsl/data/atlases/MNI/MNI-maxprob-thr25-{thickness}mm.nii.gz")
        jhu_atlas           = nib.load(f"/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-{thickness}mm.nii.gz")
        self.affine         = mni_atlas.affine
        self.mni_data       = mni_atlas.get_fdata()
        self.jhu_data       = jhu_atlas.get_fdata()
        self.mni_regions    = {"caudate": [1], "frontal lobe": [3], "insula": [4], 
        "occipital lobe": [5], "putamen": [7], "temporal lobe": [8]} 
        # see /usr/local/fsl/data/atlases/MNI.xml
        self.jhu_regions    = {"internal capsule": [17, 18, 19, 20, 21, 22]} 
        # see /usr/local/fsl/data/atlases/JHU-labels.xml
        
    def get_region(self, scan: torch.Tensor, region: str, side: str = "both"):
        n_dims = len(scan.shape)
        region = region.lower()
        cp     = scan.detach().clone()
        if n_dims in (3, 4): # single scan: C, Z, H, W
            cp = scan.squeeze()
            if region in self.mni_regions:
                cp[~ np.isin(self.mni_data,self.mni_regions[region])] = 0
            elif region in self.jhu_regions:
                cp[~ np.isin(self.jhu_data,self.jhu_regions[region])] = 0
            else:
                assert False, f"BrainRegions.get_region: Unknown region {region}"
            if side == "left":
                cp[cp.shape[0]//2:,:,:] = 0
            elif side == "right":
                cp[:cp.shape[0]//2,:,:] = 0
            elif side != "both":
                assert False, "BrainRegions.get_region: side must be 'left', 'right' or 'both'"
        elif n_dims == 5: # batch of scans
            cp = [self.get_region(cp[i,:,:,:,:].squeeze(),
                                region = region, 
                                side = side) for i in range(scan.shape[0])]
            cp = torch.stack(cp, dim = 0).unsqueeze(dim = 1)
        else:
            assert n_dims in (3, 4, 5), f"BrainRegions.get_region: invalid scan dimensions {scan.shape}"
        return cp
        
        
if __name__ == "__main__":
    import torch
    from ct_loader_torchio import CTLoader
    TABLE_DATA  = "gravo.csv"
    DATA_DIR    = "../../../data/gravo"
    ct_loader   = CTLoader(TABLE_DATA, "NCCT", 
                        balance_test_set    = True,
                        balance_train_set   = False,
                        data_dir            = DATA_DIR)
                        
    train, validation, test = ct_loader.subject_dataset()
    test_loader = torch.utils.data.DataLoader(test, 
                            batch_size  = 2, 
                            num_workers = 0,
                            pin_memory  = torch.cuda.is_available())
    br          = BrainRegions(thickness = 2)
    for subjects in test_loader:
        scans = subjects["ct"]["data"]
        for region in ("internal capsule",):
            fl    = br.get_region(scans, region)
            for i in range(fl.shape[0]):
                nib.save( nib.Nifti1Image(fl[i].squeeze().numpy(), affine=br.affine), f"{region} {i}.nii")
        break
    
