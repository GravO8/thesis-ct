import torch
from ct_loader import CTLoader
from utils.trainer import Trainer
from models.models import Custom3DCNNv1
from models.model import Baseline3DCNN

if __name__ == "__main__":
    ct_loader   = CTLoader(data_dir = "../../../data/gravo")
    trainer     = Trainer(ct_loader)
    
    to_test     = [ Baseline3DCNN(Custom3DCNNv1()), 
                    Baseline3DCNN(resnet_3d(18)),
                    Baseline3DCNN(resnet_3d(34))]
    
    
