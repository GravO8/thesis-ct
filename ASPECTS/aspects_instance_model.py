import torch

class ASPECTSInstanceModel(torch.nn.Module):
    def __init__(self, mlp_layers: list, return_probs: bool = False):
        super().__init__()
        self.return_probs = return_probs
        self.mlp = self.create_mlp(mlp_layers, return_probs)
        
    def create_mlp(self, mlp_layers: list, return_probs: bool):
        assert mlp_layers[-1] == 1
        layers = [torch.nn.Linear(mlp_layers[0], mlp_layers[1])]
        for i in range(1, len(mlp_layers)-1):
            layers.append( torch.nn.ReLU(inplace = True) )
            layers.append( torch.nn.Linear(mlp_layers[i], mlp_layers[i+1]) )
        if return_probs:
            layers.append( torch.nn.Sigmoid() )
        return torch.nn.Sequential( *layers )
        
    def __call__(self, instance):
        x = self.mlp(instance)
        if not self.return_probs:
            x = torch.sign(x)
            x = torch.relu(x)
        return x
