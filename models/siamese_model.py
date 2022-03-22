import torch
from .mlp import MLP

class SiameseNet(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, dropout: float = None, 
        mlp_layers: list = [512, 128], return_features: bool = False, 
        legacy: bool = False):
        '''
        TODO
        '''
        super(SiameseNet, self).__init__()
        if dropout is not None:
            assert len(mlp_layers) > 0, "SiameseNet.__init__: dropout can only be used in the MLP layers"
        self.encoder = encoder # CNN input is NCZHW
        if legacy:
            if dropout is None:
                self.mlp    = torch.nn.Sequential(  torch.nn.LazyLinear(512),
                                                    torch.nn.GELU(),
                                                    torch.nn.Linear(512, 256),
                                                    torch.nn.GELU(),
                                                    torch.nn.Linear(256, 1),
                                                    torch.nn.Sigmoid())
            else:
                self.mlp    = torch.nn.Sequential(  torch.nn.LazyLinear(512),
                                                    torch.nn.GELU(),
                                                    torch.nn.Dropout(p = dropout),
                                                    torch.nn.Linear(512, 256),
                                                    torch.nn.GELU(),
                                                    torch.nn.Dropout(p = dropout),
                                                    torch.nn.Linear(256, 1),
                                                    torch.nn.Sigmoid())
        else:
            self.mlp = MLP(mlp_layers, dropout = dropout, return_features = return_features, n_out = 1)
    
    def forward(self, x1, x2):
        encoding1   = self.encoder(x1)
        encoding2   = self.encoder(x2)
        diff        = torch.abs(encoding1 - encoding2)
        return self.mlp( diff )
        
    def to_dict(self):
        return {"type": "SiameseNet", 
                "specs": {"encoder": self.encoder.to_dict(), 
                              "mlp": self.mlp.to_dict()}}
