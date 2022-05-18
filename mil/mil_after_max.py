import torch, sys
sys.path.append("..")
from models.encoder import Encoder
from models.models import get_timm_model
from models.mil import MILEncoder, MILNetAfter, MaxMILPooling
from utils.main import main

def mil_after_max(encoder):
    assert encoder.global_pool is not None
    max_pooling = MaxMILPooling()
    mil_encoder = MILEncoder(encoder = encoder, mil_pooling = max_pooling)
    return MILNetAfter(mil_encoder)
    

if __name__ == "__main__":
    to_test = [mil_after_max(get_timm_model("resnet18", global_pool = "gap"))]
    main(to_test)
