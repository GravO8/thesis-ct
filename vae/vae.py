import sys
sys.path.append("..")
from utils.half_ct_loader import HalfCTLoader
from vae_trainer import VAETrainer
from utils.main import set_home, train
from models.vae import vae_v2, vae_v3


if __name__ == "__main__":
    dir     = set_home(2)
    loader  = HalfCTLoader(data_dir = dir, pad = (64, 128, 128))
    trainer = VAETrainer(loader)
    to_test = [vae_v2(), vae_v3()]
    
    train(to_test, 3, trainer)
