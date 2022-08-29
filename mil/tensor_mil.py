import torch, sys
sys.path.append("..")
from models.encoder import Encoder
from models.models import get_timm_model, custom_2D_cnn_v1
from models.mil import MILEncoder, MILTensorAxial, MaxMILPooling, MeanMILPooling, AttentionMILPooling
from utils.main import main

class TensorEncoder(torch.nn.Module):
    def __init__(self, model_name: str, out_channels: int, global_pool: str = None, 
    dim: int = 2):
        super().__init__()
        self.model_name   = model_name
        self.out_channels = out_channels
        self.global_pool  = global_pool
        self.dim          = dim
    def get_name(self):
        return f"{self.model_name}_{'features' if self.global_pool is None else self.global_pool}_{self.dim}D"
    def forward(self, x):
        x = x.squeeze()
        assert self.out_channels == x.shape[1], f"Encoder.forward: expected {self.out_channels} out features, got {x.shape[1]}."
        return x

def mil_after_max(encoder):
    assert encoder.global_pool is not None
    max_pooling       = MaxMILPooling()
    feature_extractor = Encoder("1Linear", torch.nn.Sequential(torch.nn.Linear(512,128), torch.nn.ReLU(inplace = True)), out_channels = 128, dim = 1)
    mil_encoder       = MILEncoder(encoder = encoder, mil_pooling = max_pooling, feature_extractor = feature_extractor)
    return MILTensorAxial(mil_encoder)
    
def mil_after_mean(encoder):
    assert encoder.global_pool is not None
    mean_pooling      = MeanMILPooling()
    feature_extractor = Encoder("1Linear", torch.nn.Sequential(torch.nn.Linear(512,128), torch.nn.ReLU(inplace = True)), out_channels = 128, dim = 1)
    mil_encoder       = MILEncoder(encoder = encoder, mil_pooling = mean_pooling, feature_extractor = feature_extractor)
    return MILTensorAxial(mil_encoder)
    
def mil_after_attention(encoder):
    assert encoder.global_pool is not None
    attn_pooling      = AttentionMILPooling(in_channels = encoder.out_channels)
    feature_extractor = Encoder("1Linear", torch.nn.Sequential(torch.nn.Linear(512,128), torch.nn.ReLU(inplace = True)), out_channels = 128, dim = 1)
    mil_encoder       = MILEncoder(encoder = encoder, mil_pooling = attn_pooling, feature_extractor = feature_extractor)
    return MILTensorAxial(mil_encoder)

if __name__ == "__main__":
    to_test = [ 
                mil_after_max(TensorEncoder("resnet18", 512, "gap")),
                mil_after_mean(TensorEncoder("resnet18", 512, "gap")),
                mil_after_attention(TensorEncoder("resnet18", 512, "gap"))
                ]
    main(to_test, N = 5, device = 1, from_tensors = True, skip_slices = 2)
