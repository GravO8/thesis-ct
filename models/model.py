import torch
from abc import ABC, abstractmethod
from final_mlp import FinalMLP
from encoder import Encoder

class Baseline3DCNN(Model):
    def process_input(self, x):
        return self.default_process_input(x)
    def name_appendix(self):
        return "baseline-3DCNN"

class Model(ABC, torch.nn.Module):
    def __init__(self, encoder: Encoder):
        torch.nn.Module.__init__(self)
        self.encoder    = encoder
        self.mlp        = FinalMLP(self.encoder.out_features)
        
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
