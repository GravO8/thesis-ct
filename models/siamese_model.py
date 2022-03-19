import torch

class SiameseNet(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, dropout: float = None, 
        mlp_layers: list = [518, 128], return_features: bool = False):
        '''
        TODO
        '''
        super(SiameseNet, self).__init__()
        if dropout is not None:
            assert len(mlp_layers) > 0, "SiameseNet.__init__: dropout can only be used in the MLP layers"
        self.encoder            = encoder # CNN input is NCZHW
        self.dropout            = dropout
        self.mlp_layers         = mlp_layers
        self.return_features    = return_features
        self.mlp                = self.get_mlp()
                                            
    def get_mlp(self):
        '''
        TODO
        '''
        n_layers = len(self.mlp_layers)
        layers   = []
        if n_layers > 0:
            layers.append( torch.nn.LazyLinear(self.mlp_layers[0]) )
            layers.append( torch.nn.GELU() )
            for i in range(n_layers-1):
                if self.dropout is not None:
                    layers.append( torch.nn.Dropout(p = self.dropout) )
                layers.append( torch.nn.Linear(self.mlp_layers[i], self.mlp_layers[i+1]) )
                layers.append( torch.nn.GELU() )
            if not self.return_features:
                layers.append( torch.nn.Linear(self.mlp_layers[-1], 1) )
                layers.append( torch.nn.Sigmoid() )
        else:
            if self.return_features:
                layers.append( torch.nn.Identity() )
            else:
                layers.append( torch.nn.LazyLinear(1) )
                layers.append( torch.nn.Sigmoid() )
        self.mlp = torch.nn.Sequential( *layers )
    
    def forward(self, x1, x2):
        encoding1   = self.encoder(x1)
        encoding2   = self.encoder(x2)
        diff        = torch.abs(encoding1 - encoding2)
        out         = self.mlp( diff )
        if not self.return_features:
            y_prob  = out.squeeze()
            y_pred  = torch.ge(y_prob, 0.5).float()
            return y_prob, y_pred
        return out
        
    def to_dict(self):
        return {"type": "SiameseNet", 
                "specs": {"encoder": self.encoder.to_dict(), 
                              "mlp": {"mlp_layers": self.mlp_layers,
                                      "dropout": "no" if self.dropout is None else self.dropout,
                                      "return_features": self.return_features}
                        }
                }
