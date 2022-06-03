import torch, sys
sys.path.append("..")
from models.resnet_3d import resnet_3d
from unsup_trainer import UnSupTrainer
from unsup_loader import UnSupCTLoader
from models.model import UnSup3DCNN
from utils.main import set_home, train

def unsup_main(to_test: list, N: int = 3, device: int = 3):
    dir     = set_home()
    loader  = UnSupCTLoader(data_dir = dir)
    trainer = UnSupTrainer(loader, batch_size = 32)
    train(to_test, N, trainer)


if __name__ == "__main__":
    to_test = [ UnSup3DCNN(resnet_3d(18, global_pool = "gmp")),
                UnSup3DCNN(resnet_3d(34, global_pool = "gmp")),
                UnSup3DCNN(resnet_3d(50, global_pool = "gmp"))]
    unsup_main(to_test)
