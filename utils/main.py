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
    
    
def train(to_test: list, N: int, trainer):
    for model in to_test:
        for i in range(N):
            trainer.train(model)
    print("done")


def main(to_test: list, N: int = 3, device: int = 3, slice: str = None, 
    slice_range: int = None, pad: int = None, from_tensors = False, **kwargs):
    dir = set_home(device)
    if from_tensors:
        ct_loader = CTLoaderTensors(data_dir = dir, **kwargs)
    elif slice_range is not None:
        assert slice is not None
        ct_loader = CTLoader2D(slice, slice_range = slice_range, data_dir = dir, 
                    pad = pad, **kwargs)
        for model in to_test:
            model.set_slice_info(slice_range, slice)
    else:
        ct_loader = CTLoader(data_dir = dir, **kwargs)
    trainer = Trainer(ct_loader, batch_size = 32)
    train(to_test, N, trainer)
    
