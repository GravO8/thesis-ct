import torch


class Encoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, model_name: str, out_features: int):
        torch.nn.Module.__init__(self)
        self.encoder      = encoder
        self.name         = model_name
        self.out_features = out_features
        
    def get_name(self):
        return self.name
        
    def forward(self, x):
        return self.encoder(x)
