import torch, sys
sys.path.append("..")
from utils.ct_loader import CTLoader
from models.deep_sym_net import deep_sym_encoder
from models.resnet_3d import resnet_3d
from utils.trainer import Trainer
from models.models import custom_3D_cnn_v1
from models.model import Baseline3DCNN

if __name__ == "__main__":
    ct_loader   = CTLoader(data_dir = "../../../data/gravo")
    trainer     = Trainer(ct_loader)
    
    to_test     = [ Baseline3DCNN(custom_3D_cnn_v1(global_pool = "gap")),
                    Baseline3DCNN(resnet_3d(18, global_pool = "gap")),
                    Baseline3DCNN(resnet_3d(34, global_pool = "gap")),
                    Baseline3DCNN(deep_sym_encoder(1, global_pool = "gap"))]
                    
    for model in to_test:
        for i in range(3):
            trainer.train(model)
    
    
