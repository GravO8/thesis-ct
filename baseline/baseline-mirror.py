import torch, sys
sys.path.append("..")
from models.deep_sym_net import deep_sym_encoder
from models.resnet_3d import resnet_3d
from utils.ct_loader import CTLoader
from utils.trainer import Trainer
from models.models import custom_3D_cnn_v1
from models.model import BaselineMirror

if __name__ == "__main__":
    if torch.cuda.is_available():
        dir = "/media/avcstorage/gravo"
        torch.cuda.set_device(3)
    else:
        dir = "../../../data/gravo"
    ct_loader   = CTLoader(data_dir = dir)
    trainer     = Trainer(ct_loader, batch_size = 32)
    
    to_test     = [ BaselineMirror(custom_3D_cnn_v1(global_pool = "gap")),
                    BaselineMirror(resnet_3d(18, global_pool = "gap")),
                    BaselineMirror(resnet_3d(34, global_pool = "gap")),
                    BaselineMirror(deep_sym_encoder(1, global_pool = "gap"))]
                    
    for model in to_test:
        for i in range(3):
            trainer.train(model)
    print("done")
    
    
