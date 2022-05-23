import torch
from .ct_loader import CTLoader, CTLoader2D
from .trainer import Trainer

def main(to_test: list, N: int = 3, device: int = 3, slice: str = None, 
    slice_range: int = None):
    if torch.cuda.is_available():
        dir = "/media/avcstorage/gravo"
        torch.cuda.set_device(device)
    else:
        dir = "../../../data/gravo"
    
    if slice_range is not None:
        assert slice is not None
        ct_loader = CTLoader2D(slice, slice_range = slice_range, data_dir = dir)
        for model in to_test:
            model.set_slice_range(slice_range)
    else:
        ct_loader = CTLoader(data_dir = dir)
    trainer = Trainer(ct_loader, batch_size = 32)

    for model in to_test:
        for i in range(N):
            trainer.train(model)
    print("done")
