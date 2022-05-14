import torch
from ct_loader import CTLoader
from utils.trainer import Trainer
from models.models import custom_3D_cnn_v1
from models.model import Baseline3DCNN

if __name__ == "__main__":
    ct_loader   = CTLoader(data_dir = "../../../data/gravo")
    trainer     = Trainer(ct_loader)
    
    to_test     = [Baseline3DCNN(custom_3D_cnn_v1(), 64, "customCNN")}
    
    
