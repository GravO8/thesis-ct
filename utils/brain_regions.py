import nibabel as nib

class BrainRegions:
    def __init__(self, thickness: int = 1):
        assert thickness in (1, 2), f"BrainRegions.__init__: invalid thickness. Valid values: 1 or 2."
        mni_atlas           = nib.load(f"/usr/local/fsl/data/atlases/MNI/MNI-maxprob-thr25-{thickness}mm.nii.gz")
        jhu_atlas           = nib.load(f"JHU-ICBM-labels-{thickness}mm.nii.gz")
        self.mni_data       = self.mni_atlas.get_fdata()
        self.jhu_data       = self.jhu_atlas.get_fdata()
        self.mni_regions    = {"caudate": [0], "frontal lobe": [2], "insula": [3], 
        "occipital Lobe": [4], "putamen": [6], "temporal lobe": [7]} 
        # see /usr/local/fsl/data/atlases/MNI.xml
        self.jhu_regions    = {"internal capsule": [17, 18, 19, 20, 21, 22]} 
        # see /usr/local/fsl/data/atlases/JHU-labels.xml
        
    def get_region(self, scan, region: str, side: str = "both"):
        n_dims = len(scan.shape)
        region = region.lower()
        if n_dims in (3, 4): # single scan: C, Z, H, W
            cp = scan.copy().squeeze()
            if region in self.mni_regions:
                cp[~ self.mni_data.isin(self.mni_regions[region])] = 0
            elif region in self.jhu_regions:
                cp[~ self.jhu_data.isin(self.jhu_regions[region])] = 0
            else:
                assert False, f"BrainRegions.get_region: Unknown region {region}"
            if side == "left":
                cp[cp.shape[0]//2:,:,:] = 0
            elif side == "right":
                cp[:cp.shape[0]//2,:,:] = 0
            elif side != "all":
                assert False, "BrainRegions.get_region: side must be 'left', 'right' or 'both'"
        elif n_dims == 5: # batch of scans
            cp = [self.get_region(scan[i,:,:,:,:].squeeze(),
                                region = region, 
                                side = side) for i in range(len(scan.shape[0]))]
            torch.stack(cp, dim = 0) 
        else:
            assert n_dims in (3, 4, 5), f"BrainRegions.get_region: invalid scan dimensions {scan.shape}"
        return cp
