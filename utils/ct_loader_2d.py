import torchio, torch, numpy, os
from .ct_loader_torchio import CTLoader


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
    
    def set_mask(self, debug = False):
        '''
        TODO
        '''
        dirs        = [os.path.join(self.data_dir, self.ct_type, f"{patient_id}.nii") for patient_id in self.scan_ids]
        slices      = [torchio.ScalarImage(dirs[i])[torchio.DATA][:,:,:,self.scan_slices[i]].squeeze() for i in range(len(dirs))]
        self.mask   = slices[0]
        for i in range(1, len(slices)-1):
            self.mask += slices[i]
        self.mask          /= len(slices)
        self.negative_mask  = 1 - self.mask
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(self.mask.squeeze().T.flip(0), cmap = "gray")
            plt.show()
            plt.imshow(self.negative_mask.squeeze().T.flip(0), cmap = "gray")
            plt.show()
    
    def create_subject(self, row, transform: str = None, debug = False):
        '''
        TODO
        '''
        subject = super().create_subject(row, transform = transform)
        scan    = subject["ct"][torchio.DATA]
        scores  = []
        for i in range(scan.shape[-1]):
            ax_slice    = scan[:,:,:,i].squeeze() # shape = (B,x,y,z)
            score       = (ax_slice*self.mask).sum() - (ax_slice*self.negative_mask).sum()
            scores.append(score)
        i     = numpy.argmax(scores)
        scan  = scan[:,:,:,i-self.slice_interval:i+self.slice_interval].mean(axis = 3).unsqueeze(dim = 0)
        scan  = torch.nn.functional.interpolate(scan, scale_factor = 2, mode = "bilinear", antialias = True)
        scan  = scan.squeeze(dim = 0)
        if self.pad is not None:
            _, w, h = scan.shape
            zeros   = torch.zeros(1, self.pad, self.pad)
            pad_w   = (self.pad-w)//2
            pad_h   = (self.pad-h)//2
            zeros[:, pad_w:pad_w+w, pad_h:pad_h+h] = scan
            scan = zeros
        subject["ct"][torchio.DATA] = torch.Tensor(scan.unsqueeze(len(scan.shape)))
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(scan.squeeze().T.flip(0), cmap = "gray")
            plt.show()
        return subject
        
    def subject_dataset(self, train_size: float = 0.75):
        '''
        TODO
        '''
        train_loader, validation_loader, test_loader = super().subject_dataset(train_size = train_size)
        mask_leak = False
        for subject in train_loader:
            patient_id = subject["patient_id"]
            if (subject["transform"] == "original") and (patient_id in self.scan_ids):
                print(f"CNNTrainer2D.subject_dataset: patient {patient_id} used for mask is not on train set")
                mask_leak = True
        # assert not mask_leak, "CNNTrainer2D.subject_dataset: scans used to obtain mask are not in the train set"
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
        dict = super().to_dict()
        dict["scan_ids"]        = self.scan_ids
        dict["scan_slices"]     = self.scan_slices
        dict["slice_interval"]  = self.slice_interval
        dict["pad"]             = self.pad
        return dict
