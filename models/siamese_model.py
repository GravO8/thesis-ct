import torch
from .mlp import MLP

class SiameseNet(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, dropout: float = None, 
        mlp_layers: list = [512, 128, 1], return_features: bool = False):
        '''
        TODO
        '''
        super(SiameseNet, self).__init__()
        if dropout is not None:
            assert len(mlp_layers) > 0, "SiameseNet.__init__: dropout can only be used in the MLP layers"
        self.encoder            = encoder # CNN input is NCZHW
        self.return_features    = return_features
        self.mlp                = MLP(mlp_layers, dropout = dropout, return_features = return_features)
    
    def forward(self, x1, x2):
        encoding1   = self.encoder(x1)
        encoding2   = self.encoder(x2)
        diff        = torch.abs(encoding1 - encoding2)
        return self.mlp( diff )
        
    def to_dict(self):
        return {"type": "SiameseNet", 
                "specs": {"encoder": self.encoder.to_dict(), 
                              "mlp": self.mlp.to_dict()}}
