import torch
from .ct_loader import CTLoader, CTLoader2D, CTLoaderTensors
from .trainer import Trainer

def set_home(device: int):
    if torch.cuda.is_available():
        dir = "/media/avcstorage/gravo"
        torch.cuda.set_device(device)
    else:
        dir = "../../../data/gravo"
    return dir
    
    
def train(to_test: list, ct_loader, trainer, N: int = None):
    if ct_loader.kfold:
        for model in to_test:
            for train, test in ct_loader.get_folds():
                trainer.train(model, train, test, ct_loader.skip_slices)
    else:
        assert N is not None
        for model in to_test:
            for _ in range(N):
                ct_loader.reshuffle()
                train, test = ct_loader.load_dataset()
                trainer.train(model, train, test, ct_loader.skip_slices)
    print("done")


def main(to_test: list, device: int = 3, slice: str = None, 
    slice_range: int = None, pad: int = None, from_tensors = False, N: int = None, 
    kfold = False, **kwargs):
    dir = set_home(device)
    if from_tensors:
        ct_loader = CTLoaderTensors(data_dir = dir, kfold = kfold, **kwargs)
    elif slice_range is not None:
        assert slice is not None
        ct_loader = CTLoader2D(slice, kfold = kfold, slice_range = slice_range, data_dir = dir, 
                    pad = pad, **kwargs)
        for model in to_test:
            model.set_slice_info(slice_range, slice)
    else:
        ct_loader = CTLoader(data_dir = dir, kfold = kfold, **kwargs)
    trainer = Trainer(batch_size = 32)
    train(to_test, ct_loader, trainer, N)
    
