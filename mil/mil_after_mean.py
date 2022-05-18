import torch, sys
sys.path.append("..")
from models.encoder import Encoder
from models.models import get_timm_model, custom_2D_cnn_v1
from models.mil import MILEncoder, MILNetAfter, MeanMILPooling
from utils.main import main

def mil_after_mean(encoder):
    assert encoder.global_pool is not None
    max_pooling = MeanMILPooling()
    mil_encoder = MILEncoder(encoder = encoder, mil_pooling = max_pooling)
    return MILNetAfter(mil_encoder)
    

if __name__ == "__main__":
    to_test = [ mil_after_mean(custom_2D_cnn_v1(global_pool = "gap")),
                mil_after_mean(get_timm_model("resnet18", global_pool = "gap")),
                mil_after_mean(get_timm_model("resnet34", global_pool = "gap")),
                mil_after_mean(get_timm_model("efficientnet_b0", global_pool = "gap")),
                mil_after_mean(get_timm_model("efficientnet_b1", global_pool = "gap"))]
    main(to_test)
