import torch, json, os

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
        self.init_weights()
        
    def init_weights(self):
        models      = [file for file in os.listdir("weights") if (file.endswith(".json") and file.startswith("mlp-"))]
        new_weights = True
        for model in models:
            if self.same_mlp(model):
                self.load_weights(model)
                new_weights = False
        if new_weights:
            mlp_name = f"weights/mlp-{len(models)}"
            torch.save(self.mlp.state_dict(), f"{mlp_name}.pt")
            with open(f"{mlp_name}.json", "w") as f:
                json.dump(self.to_dict(), f, indent = 4) 
                
    def same_mlp(self, other_model: str):
        self_model = self.to_dict()
        with open(f"weights/{other_model}") as json_file:
            other_model = json.load(json_file)
        return ((self_model["layers_list"] == other_model["layers_list"]) and
                (self_model["dropout"] == other_model["dropout"]) and
                (self_model["return_features"] == other_model["return_features"]))
                
    def load_weights(self, model_name: str):
        weights = f"weights/{model_name[:-5]}.pt"
        if torch.cuda.is_available():
            self.mlp.load_state_dict( torch.load(weights) )
        else:
            self.mlp.load_state_dict( torch.load(weights, map_location = torch.device("cpu")) )
        
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


if __name__ == "__main__":
    mlp = MLP(layers_list = [30, 20, 1], dropout = .2, return_features = False, hidden_activation = torch.nn.ReLU())
    
