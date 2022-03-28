import torch

class MLP(torch.nn.Module):
    def __init__(self, layers_list: list, dropout: float = None, 
    return_features: bool = False, hidden_activation = torch.nn.GELU()):
        '''
        TODO
        layers_list: list of integers
        '''
        super(MLP, self).__init__()
        assert len(layers_list) > 0, "MLP.__init__: MLP must have at least 1 layer"
        self.layers_list        = layers_list
        self.dropout            = dropout
        self.return_features    = return_features
        self.hidden_activation  = hidden_activation
        self.set_mlp()
        
    def set_mlp(self):
        '''
        TODO
        '''
        n_layers = len(self.layers_list)
        layers   = []
        layers.append( torch.nn.LazyLinear(self.layers_list[0]) )
        for i in range(n_layers-1):
            layers.append( self.hidden_activation )
            if self.dropout is not None:
                layers.append( torch.nn.Dropout(p = self.dropout) )
            layers.append( torch.nn.Linear(self.layers_list[i], self.layers_list[i+1]) )
        if self.return_features:
            layers.append( self.hidden_activation )
        else:
            if self.layers_list[-1] == 1:
                layers.append( torch.nn.Sigmoid() )
            else:
                layers.append( torch.nn.Softmax() )
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
                "hidden_activation": self.hidden_activation.__class__.__name__}
