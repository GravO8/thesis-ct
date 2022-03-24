import torch

class MLP(torch.nn.Module):
    def __init__(self, layers_list: list, dropout: float = None, 
    return_features: bool = False, n_out: int = -1, hidden_activation = torch.nn.GELU()):
        '''
        TODO
        layers_list: list of integers
        '''
        super(MLP, self).__init__()
        self.layers_list        = layers_list
        self.dropout            = dropout
        self.return_features    = return_features
        self.n_out              = n_out
        self.hidden_activation  = hidden_activation
        if not self.return_features:
            assert self.n_out != -1, "MLP.__init__: set the number of output values with the 'n_out' parameter"
        self.set_mlp()
        
    def set_mlp(self):
        '''
        TODO
        '''
        n_layers = len(self.layers_list)
        layers   = []
        if n_layers > 0:
            layers.append( torch.nn.LazyLinear(self.layers_list[0]) )
            layers.append( self.hidden_activation )
            for i in range(n_layers-1):
                if self.dropout is not None:
                    layers.append( torch.nn.Dropout(p = self.dropout) )
                layers.append( torch.nn.Linear(self.layers_list[i], self.layers_list[i+1]) )
                layers.append( self.hidden_activation )
            if not self.return_features:
                if self.dropout is not None:
                    layers.append( torch.nn.Dropout(p = self.dropout) )
                layers.append( torch.nn.Linear(self.layers_list[-1], self.n_out) )
                if self.n_out > 1:  layers.append( torch.nn.Softmax() )
                else:               layers.append( torch.nn.Sigmoid() )
        else:
            if self.return_features:
                layers.append( torch.nn.Identity() )
            else:
                layers.append( torch.nn.LazyLinear(self.n_out) )
                if self.n_out > 1:  layers.append( torch.nn.Softmax() )
                else:               layers.append( torch.nn.Sigmoid() )
        self.mlp = torch.nn.Sequential( *layers )
        
    def forward(self, x):
        out = self.mlp(x)
        if self.return_features:
            return torch.nn.functional.normalize(out, p = 2, dim = 1)
        return out
        
    def to_dict(self):
        return {"layers_list": self.layers_list,
                "dropout": self.dropout,
                "return_features": self.return_features,
                "n_out": self.n_out,
                "hidden_activation": self.hidden_activation.__class__.__name__}
