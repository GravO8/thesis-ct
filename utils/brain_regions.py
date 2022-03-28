import nibabel as nib

class BrainRegions:
    def __init__(self):
        self.mni_atlas = nib.load("/usr/local/fsl/data/atlases/MNI/MNI-maxprob-thr25-1mm.nii.gz")
        
