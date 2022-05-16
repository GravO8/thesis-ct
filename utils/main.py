import torch
from .ct_loader import CTLoader
from .trainer import Trainer

def main(to_test, N = 3, device = 3):
    if torch.cuda.is_available():
        dir = "/media/avcstorage/gravo"
        torch.cuda.set_device(device)
    else:
        dir = "../../../data/gravo"
    ct_loader   = CTLoader(data_dir = dir)
    trainer     = Trainer(ct_loader, batch_size = 32)
                    
    for model in to_test:
        for i in range(N):
            trainer.train(model)
    print("done")
