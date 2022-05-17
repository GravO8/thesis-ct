import torch
from .encoder import Encoder
from .model import Model

class MILPooling(torch.nn.Module):
    pass
    
class MILEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, mil_pooling: MILPooling):
        torch.nn.Module.__init__(self)
        self.encoder      = encoder # instance encoder
        self.mil_pooling  = mil_pooling
        self.out_channels = self.encoder.out_channels
    def get_name(self):
        return f"{self.encoder.get_name()}_{self.mil_pooling.get_name()}"
    def forward(self, x):
        shp = x.shape # (C,x,y,z) = (1,x,y,z)
        assert len(shp) == 4, "MILEncoder.forward: can only process one CT scan at the time."
        x.reshape((shp[3],1,shp[1],shp[2])) # (B,C,x,y) = (z,1,x,y)
        1/0 # check if the axial slices are correct
        x = self.encoder(x)
        x = self.mil_pooling(x)
        return x
        
