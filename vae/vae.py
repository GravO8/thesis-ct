import sys, torch
sys.path.append("..")
from utils.half_ct_loader import HalfCTLoader
from vae_trainer import VAETrainer
from utils.main import set_home, train
from models.vae import vae_v1


if __name__ == "__main__":
    dir     = set_home(0)
    shape   = (64, 128, 128)
    loader  = HalfCTLoader(data_dir = dir, pad = shape)
    trainer = VAETrainer(loader)
    model   = vae_v1(shape = shape, n_start_chans = 8, N = 6)
    
    train([model], 3, trainer)
