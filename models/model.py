import torch
from abc import ABC, abstractmethod
from models import final_mlp
from encoder import Encoder, SiameseEncoder
        
        
class Model(ABC, torch.nn.Module):
    def __init__(self, encoder: Encoder):
        torch.nn.Module.__init__(self)
        self.encoder    = encoder
        self.mlp        = final_mlp(self.encoder.out_features)
        
    @abstractmethod
    def process_input(self, x):
        pass
        
    @abstractmethod
    def name_appendix():
        pass
        
    def default_process_input(self, x):
        return x/100
    
    def forward(self, x):
        x = self.process_input(x)
        x = self.encoder(x)
        return self.mlp(x)
        
    def get_name(self):
        return self.name_appendix() + "-" + self.encoder.get_name()


class Baseline3DCNN(Model):
    def process_input(self, x):
        return self.default_process_input(x)
    def name_appendix(self):
        return "baseline-3DCNN"
        
        
class SiameseNet(Model):
    def __init__(self, encoder: SiameseEncoder):
        assert isinstance(encoder, SiameseEncoder), "SiameseNet.__init__: 'encoder' must be of class 'SiameseEncoder'"
        super().__init__(encoder)
    def process_input(self, x):
        x           = self.default_process_input(x)
        msp         = x.shape[2]//2             # midsagittal plane
        hemisphere1 = x[:,:,:msp,:,:]           # shape = (B,C,x,y,z)
        hemisphere2 = x[:,:,msp:,:,:].flip(2)   # B - batch; C - channels
        return (hemisphere1, hemisphere2)
    def name_appendix(self):
        return "SiameseNet"
