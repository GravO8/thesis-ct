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
        assert x.dim() == 2, f"MILPoolingEncodings.__call__: expected tensor of size 2 (number of instances, instance encoding size) but got {x.dim()}"
        return self.forward(x)
    @abstractmethod
    def forward(self, x):
        pass
    
class MaxMILPooling(MILPoolingEncodings):
    def forward(self, x):
        return x.max(dim = 0).values
    def get_name(self):
        return "MaxPooling"
        
class MeanMILPooling(MILPoolingEncodings):
    def forward(self, x):
        return h.mean(dim = 0)
    def get_name(self):
        return "MeanPooling"
        
class AttentionMILPooling(MILPoolingEncodings, torch.nn.Module):
    def __init__(self, in_channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attention = torch.nn.Sequential(
                            torch.nn.Linear(in_channels, bottleneck_dim),
                            torch.nn.Tanh(),
                            torch.nn.Linear(bottleneck_dim, 1))
    def forward(self, x):
        a = self.attention(x)
        a = torch.nn.Softmax(dim = 0)(a).T
        x = torch.mm(a,x)
        return x
    def get_name(self):
        return "AttentionPooling"


class MILEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, mil_pooling: MILPooling, feature_extractor: Encoder = None):
        torch.nn.Module.__init__(self)
        self.encoder           = encoder # instance encoder
        self.mil_pooling       = mil_pooling
        self.feature_extractor = feature_extractor
        self.out_channels      = self.encoder.out_channels
    def get_name(self):
        return f"{self.encoder.get_name()}_{self.mil_pooling.get_name()}" + ("" if self.feature_extractor is None else "_"+self.feature_extractor.get_name())
    def forward(self, x):
        x = self.encoder(x)
        x = self.mil_pooling(x)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return x
        
        
class MILNet(Model):
    def __init__(self, encoder: MILEncoder):
        assert isinstance(encoder, MILEncoder), "MILNet.__init__: 'encoder' must be of class 'MILEncoder'"
        super().__init__(encoder)
    def forward(self, batch):
        out = []
        for scan in batch.unbind(dim = 0):
            out.append( super().forward(scan) )
        return torch.stack(out, dim = 0)
    def process_input(self, x):
        x = self.normalize_input(x)
        x = x.permute((3,0,1,2)) # (C,x,y,z) = (1,x,y,z) -> (z,C,x,y) = (B,1,x,y)
        x = x[[i for i in range(x.shape[0]) if torch.count_nonzero(x[i,:,:,:] > 0) > 100]]
        return x
    def name_appendix(self):
        return "MILNet"
        
class MILNetAfter(MILNet):
    def __init__(self, encoder: MILEncoder):
        super().__init__(encoder)
        assert encoder.encoder.global_pool is not None
        assert (encoder.feature_extractor is None) or (encoder.feature_extractor.global_pool is None)
    def name_appendix(self):
        return super().name_appendix() + "-" + "after"
