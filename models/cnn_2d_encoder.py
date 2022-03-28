import timm, torch


class CNN2DEncoder(torch.nn.Module):
    def __init__(self, cnn_name: str = "resnet50", pretrained: bool = False, 
        n_features = "same", freeze: bool = False, drop_block_rate: float = 0.0, 
        drop_rate: float = 0.0, normalization = torch.nn.BatchNorm2d, in_channels: float = 1):
        super(CNN2DEncoder, self).__init__()
        if freeze:
            assert pretrained, "CNN2DEncoder.__init__: frozen model requires pretrained=True"
        if pretrained:
            assert (drop_block_rate != 0.0) and (drop_rate != 0.0), "CNN2DEncoder.__init__: pretrained model can't use dropout"
        self.cnn_name       = cnn_name
        self.pretrained     = pretrained
        self.n_features     = n_features
        self.freeze         = freeze
        self.drop_block_rate    = drop_block_rate
        self.drop_rate          = drop_rate
        self.normalization      = normalization.__class__.__name__
        self.resnet             = timm.create_model(cnn_name, 
                                                    pretrained = pretrained, 
                                                    in_chans = in_channels,
                                                    drop_block_rate = drop_block_rate,
                                                    drop_rate = drop_rate,
                                                    norm_layer = normalization)
        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        if n_features == "same":
            self.resnet.fc = torch.nn.Identity()
        elif isinstance(n_features, int) or n_features.isnumeric():
            self.resnet.fc = nn.LazyLinear(int(n_features))
        else:
            assert False
    def forward(self, x):
        return self.resnet(x)
    def to_dict(self):
        return {"cnn_name": self.cnn_name, 
                "pretrained": self.pretrained, 
                "n_features": self.n_features,
                "freeze": self.freeze,
                "drop_block_rate": self.drop_block_rate,
                "drop_rate": self.drop_rate,
                "normalization": self.normalization.__class__.__name__}


if __name__ == "__main__":
    r = CNN2DEncoder("resnet34", drop_rate = .1)
    print(r.resnet)
    # from torchsummary import summary
    # sample = torch.rand(2,1,91,180)
    # summary(r, (1,91,180))
    # r(sample)
    # for param in resnet.parameters():
    #     print( param.requires_grad )
