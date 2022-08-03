import torch

class ASPECTSInstanceModel(torch.nn.Module):
    def __init__(self, mlp_layers: list, temperature: int = 6):
        super().__init__()
        self.temperature = temperature
        self.mlp = self.create_mlp(mlp_layers)
        
    def create_mlp(self, mlp_layers: list):
        assert mlp_layers[-1] == 1
        layers = [torch.nn.Linear(mlp_layers[0], mlp_layers[1], bias = False)]
        for i in range(1, len(mlp_layers)-1):
            # layers.append( torch.nn.ReLU(inplace = True) )
            layers.append( torch.nn.Sigmoid() )
            layers.append( torch.nn.Linear(mlp_layers[i], mlp_layers[i+1], bias = False) )
        return torch.nn.Sequential( *layers )
        
    def __call__(self, instance):
        x = self.mlp(instance)
        x = x * self.temperature
        x = torch.sigmoid(x)
        return x
