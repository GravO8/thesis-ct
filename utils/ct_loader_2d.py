import torchio, torch
import nibabel as nib
from ct_loader_torchio import CTLoader


class AxialLoader(CTLoader):
    def __init__(self, table_data_file: str, ct_type, scan_ids: list, 
    scan_slices: list, slice_interval: int = 2, pad = None, **kwargs):
    '''
    TODO
    scan_ids    - list of patients ids whose scan is going to be used to create
    the mask that is used to select the exam's slices
    scan_slices - the index of the reference slice of each of the scans 
    listed in the scan_ids argument
    '''
    super(AxialLoader, self).__init__(table_data_file, ct_type, **kwargs)
    assert len(scan_ids) == len(scan_slices), "AxialLoader.__init__: scan_ids and scan_slices must have the same length"
    assert len(scan_ids) > 0, "AxialLoader.__init__: supply at least 1 reference scan"
    self.scan_ids       = scan_ids
    self.scan_slices    = scan_slices
    self.slice_interval = slice_interval
    self.pad            = pad
    self.set_mask()
    
    def set_mask(self):
        '''
        TODO
        '''
        dirs        = [os.path.join(self.data_dir, self.ct_type, f"{patient_id}.nii") for patient_id in self.scan_ids]
        slices      = [nib.load(dirs[i]).get_fdata()[:,:,self.scan_slices[i]].squeeze() for i in range(len(dirs))]
        self.mask   = slices[0]
        for i in range(1, len(slices)-1):
            self.mask += slices[i]
        self.mask  /= len(slices)
        self.negative_mask = self.mask.max() - self.mask
    
    def create_subject(self, row, transform: str = None, debug = False):
        '''
        TODO
        '''
        subject = super(CTLoader, self).create_subject(row, transform = transform)
        scan    = subject["ct"][torchio.DATA]
        scores  = []
        for i in range(scan.shape[-1]):
            ax_slice    = scan[:,:,:,i].squeeze() # shape = (B,x,y,z)
            score       = (ax_slice*self.mask).sum() - (ax_slice*self.negative_mask).sum()
            scores.append(score)
        i       = numpy.argmax(scores)
        sample  = scan[:,:,:,i-self.slice_interval:i+self.slice_interval].mean(axis = 3)
        sample  = torch.nn.functional.interpolate(sample, scale_factor = 2, mode = "bilinear", antialias = True)
        if self.pad is not None:
            W, H  = (self.pad, self.pad)
            shp   = sample.shape
            zeros = torch.zeros(1, W, H)
            w     = (W-shp[2])//2
            h     = (H-shp[3])//2
            zeros[:,:,w:w+shp[2],h:h+shp[3]] = sample
            sample = zeros
        subject["ct"] = torch.Tensor(sample)
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(sample.squee().T.flip(0), cmap = "gray")
            plt.show()
        return subject
        
    def subject_dataset(self, train_size: float = 0.75):
        '''
        TODO
        '''
        train_loader, validation_loader, test_loader = super(CTLoader, self).subject_dataset(train_size = train_size)
        mask_leak = False
        for batch in train_loader:
            for patient_id in batch["patient_id"]:
                if patient_id in self.scan_ids:
                    print(f"CNNTrainer2D.subject_dataset: patient {patient_id} used for mask is not on train set")
                    mask_leak = True
        assert not mask_leak, "CNNTrainer2D.subject_dataset: scans used to obtain mask are not in the train set"
        return train_loader, validation_loader, test_loader
        
    def k_fold(self, k: int = 5):
        '''
        TODO
        '''
        assert False
        
    def to_dict(self):
        '''
        TODO
        '''
        dict = super(CTLoader, self).to_dict()
        dict["scan_ids"]        = self.scan_ids
        dict["scan_slices"]     = self.scan_slices
        dict["slice_interval"]  = self.slice_interval
        dict["pad"]             = self.pad
        return dict
