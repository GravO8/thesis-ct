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
        
class Encoder3D(Encoder):
    def __init__(self, encoder: torch.nn.Module, model_name: str, out_features: int, 
        gap: bool = False, gmp: bool = False):
        super().__init__(encoder, model_name, out_features)
        self.gap = gap  # global average pooling
        self.gmp = gmp  # globalmax pooling
        if self.gap: assert not self.gmp
        if self.gmp: assert not self.gap
    def forward(self, x):
        x = self.encoder(x)
        if self.gap:
            x = x.mean((2,3,4))
        elif self.gmp:
            x = x.amax((2,3,4))
        return x
