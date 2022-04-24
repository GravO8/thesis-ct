import timm, torch
from .same_init_weights import SameInitWeights


class CNN2DEncoder(torch.nn.Module, SameInitWeights):
    def __init__(self, cnn_name: str = "resnet50", pretrained: bool = False, 
        n_features = "same", freeze: bool = False, drop_block_rate: float = 0.0, 
        drop_rate: float = 0.0, normalization = torch.nn.BatchNorm2d, in_channels: float = 1,
        load_local: bool = True):
        torch.nn.Module.__init__(self)
        if freeze:
            assert pretrained, "CNN2DEncoder.__init__: frozen model requires pretrained=True"
        if "efficientnet" in cnn_name:
            assert drop_block_rate == 0.0, "CNN2DEncoder.__init__: efficientnet don't have 'drop_block_rate' parameter"
        # if pretrained:
        #     assert (drop_block_rate != 0.0) and (drop_rate != 0.0), "CNN2DEncoder.__init__: pretrained model can't use dropout"
        self.cnn_name           = cnn_name
        self.pretrained         = pretrained
        self.n_features         = n_features
        self.freeze             = freeze
        self.drop_block_rate    = drop_block_rate
        self.drop_rate          = drop_rate
        self.normalization      = normalization
        self.in_channels        = in_channels
        if self.pretrained:     # if we want to load the model pretrained
            self.set_model()    # we don't want to load the local random weights
        else:
            SameInitWeights.__init__(self, load_local)  
            # must be called last because the method set_model is called by the
            # SameInitWeights constructor
        
    def set_model(self):
        kwargs = {  "pretrained":   self.pretrained,
                    "in_chans":     self.in_channels,
                    "drop_rate":    self.drop_rate,
                    "norm_layer":   self.normalization}
        if "efficientnet" not in self.cnn_name:
            kwargs["drop_block_rate"] = self.drop_block_rate
        self.encoder = timm.create_model(self.cnn_name, **kwargs)
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if self.n_features == "same":
            self.encoder.fc = torch.nn.Identity()
        elif isinstance(self.n_features, int) or self.n_features.isnumeric():
            self.encoder.fc = torch.nn.LazyLinear(int(self.n_features))
        else:
            assert False
            
    def forward(self, x):
        return self.encoder(x)
        
    def equals(self, other_model):
        return super().equals(other_model, cols = ["cnn_name", "in_channels", 
        "n_features", "drop_block_rate", "drop_rate"])
        
    def to_dict(self):
        return {"cnn_name": self.cnn_name, 
                "pretrained": self.pretrained,
                "in_channels": self.in_channels,
                "n_features": self.n_features,
                "freeze": self.freeze,
                "drop_block_rate": self.drop_block_rate,
                "drop_rate": self.drop_rate,
                "normalization": str(self.normalization)}


if __name__ == "__main__":
    r = CNN2DEncoder("resnet50", drop_rate = .1, pretrained = True)
    # print(r.to_dict())
    # print(r.encoder)
    from torchsummary import summary
    # sample = torch.rand(2,1,91,180)
    summary(r, (1,91,180))
    # r(sample)
    # for param in encoder.parameters():
    #     print( param.requires_grad )
