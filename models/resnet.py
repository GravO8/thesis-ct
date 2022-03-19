import timm, torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class ResNet(nn.Module):
    def __init__(self, version: str = "resnet50d", pretrained: bool = True, 
        n_features = "same", freeze = False):
        super(ResNet, self).__init__()
        assert not freeze or (freeze and pretrained)
        self.version    = version
        self.pretrained = pretrained
        self.n_features = n_features
        self.freeze     = freeze
        self.resnet     = timm.create_model(version, 
                                        pretrained = pretrained, 
                                        in_chans = 1)
        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        if n_features == "same":
            self.resnet.fc = Identity()
        elif isinstance(n_features, int) or n_features.isnumeric():
            self.resnet.fc = nn.LazyLinear(int(n_features))
        else:
            assert False
    def forward(self, x):
        return self.resnet(x)
    def __repr__(self):
        return f"{self.version}({'pretrained' if self.pretrained else 'scratch'},{'identity' if self.n_features == 'same' else self.n_features})"
    def to_dict(self):
        return {"version": self.version, 
                "pretrained": self.pretrained, 
                "n_features": self.n_features,
                "freeze": self.freeze}


if __name__ == "__main__":
    resnet = ResNet()
    print(resnet)
    # for param in resnet.parameters():
    #     print( param.requires_grad )
