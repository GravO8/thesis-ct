import torch, sys
sys.path.append("..")
from models.encoder import Encoder
from models.models import get_timm_model, custom_2D_cnn_v1
from models.mil import MILEncoder, MILAfterBlock, MaxMILPooling, MeanMILPooling
from utils.main import main

def mil_after_blocks(pooling):
    encoder = custom_2D_cnn_v1(global_pool = "gap")
    mil_encoder = MILEncoder(encoder = encoder, mil_pooling = pooling)
    return MILAfterBlock(mil_encoder)
    

if __name__ == "__main__":
    to_test = [ mil_after_blocks(MaxMILPooling()),
                mil_after_blocks(MeanMILPooling())]
    main(to_test)
