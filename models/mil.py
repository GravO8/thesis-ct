import torch
from .encoder import Encoder
from .model import Model
from abc import ABC, abstractmethod


class MILPooling(ABC):
    @abstractmethod
    def __call__(self, x):
        pass
    @abstractmethod
    def get_name(self):
        pass
        
class MILPoolingEncodings(MILPooling):
    def __call__(self, x):
        assert x.dim() == 2, "MILPoolingEncodings.__call__: expected tensor of size 2 (number of instances, instance encoding size)"
        self.forward(x)
    @abstractmethod
    def forward(self, x):
        pass
    
class MaxMILPooling(MILPoolingEncodings):
    def forward(self, x):
        return x.max(dim = 0).values
    def get_name(self):
        return "MaxPooling"
        
class MeanMILPooling(MILPoolingEncodings):
    def __call__(self, x):
        return h.mean(dim = 0)
    def get_name(self):
        return "MeanPooling"
        
class AttentionMILPooling(MILPoolingEncodings, torch.nn.Module):
    def __init__(self, in_channels: int, bottleneck_dim: int = 128):
        super().__init__(self)
        self.attention = torch.nn.Sequential(
                            torch.nn.Linear(in_channels, bottleneck_dim),
                            torch.nn.Tanh(),
                            torch.nn.Linear(bottleneck_dim, 1))
    def __call__(self, x):
        # verificar se isto está correto
        print("x" x.shape)
        a = self.attention(h).T
        print("a", a.shape)
        a = torch.nn.Softmax(dim = 1)(a)
        print("a Softmax", a.shape)
        x = x * a
        print("x after", x.shape)
        1/0
        return x
    
class MILEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, mil_pooling: MILPooling, feature_extractor = None: Encoder):
        torch.nn.Module.__init__(self)
        self.encoder           = encoder # instance encoder
        self.mil_pooling       = mil_pooling
        self.feature_extractor = feature_extractor
        self.out_channels      = self.encoder.out_channels
    def get_name(self):
        return f"{self.encoder.get_name()}_{self.mil_pooling.get_name()}" + ("" if self.feature_extractor is None else "_"+self.feature_extractor.get_name())
    def forward(self, x):
        shp = x.shape # (C,x,y,z) = (1,x,y,z)
        assert len(shp) == 4, "MILEncoder.forward: can only process one CT scan at the time."
        x.reshape((shp[3],1,shp[1],shp[2])) # (B,C,x,y) = (z,1,x,y)
        1/0 # check if the axial slices are correct
        x = self.encoder(x)
        x = self.mil_pooling(x)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return x
        
