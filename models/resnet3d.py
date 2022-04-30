import timm, torch
from .same_init_weights import SameInitWeights
from torchvision.ops import drop_block


class LayerNorm(torch.nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.norm = None
    def forward(self, x):
        if self.norm is None:
            _, C, Z, H, W   = x.shape
            self.norm       = torch.nn.LayerNorm([C, Z, H, W], elementwise_affine = False)
        return self.norm(x)


class ResNet3D(torch.nn.Module, SameInitWeights):
    def __init__(self, version: str = "resnet34", in_channels = 1, n_features = "same", 
    drop_rate: float = 0.0, drop_block_rate: float = 0.0, normalization = "batch",
    remove_first_maxpool: bool = False):
        torch.nn.Module.__init__(self)
        assert "resnet" in version
        assert normalization in ("batch", "layer", "group")
        assert 0 <= drop_rate < 1
        assert 0 <= drop_block_rate < 1
        self.version                = version
        self.in_channels            = in_channels
        self.n_features             = n_features
        self.drop_rate              = drop_rate
        self.drop_block_rate        = drop_block_rate
        self.normalization          = normalization
        self.remove_first_maxpool   = remove_first_maxpool
        SameInitWeights.__init__(self)  # must be called last because the
                                        # method set_model is called by the
                                        # SameInitWeights constructor

    def set_model(self):
        self.layers             = self.convert_to_3D()
        self.layers, self.fc    = self.layers[:-1], self.layers[-1]
        if self.n_features == "same":
            self.fc = torch.nn.Identity()
        elif isinstance(self.n_features, int) or self.n_features.isnumeric():
            self.fc = torch.nn.LazyLinear(int(self.n_features))
        else:
            assert False
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.drop_rate != 0.0:
            x = torch.nn.functional.dropout(x, p = float(self.drop_rate))
        x = self.fc(x)
        if self.n_features == 1:
            x = torch.sigmoid(x)
        return x
    
    def convert_to_3D(self):
        model_2d = timm.create_model(self.version, 
                                    in_chans = self.in_channels,
                                    drop_block_rate = self.drop_block_rate)
        return self.get_children( model_2d )
    
    def get_children(self, model: torch.nn.Module):
        # Adapted from: https://stackoverflow.com/a/65112132
        if list(model.children()) == []:
            return self.layer_to_3D(model)
        return self.create_multi_layer(model)

    def create_multi_layer(self, model: torch.nn.Module):
        layers  = [self.get_children(child) for child in list(model.children())]
        name    = model.__class__.__name__
        if name == "Sequential":
            return torch.nn.Sequential( *layers )
        if name == "ResNet":
            if self.remove_first_maxpool:
                del layers[3]
            return torch.nn.Sequential( *layers )
        elif (name == "BasicBlock") or (name == "Bottleneck"):
            return ResidualBlock3D(layers)
        elif name == "SelectAdaptivePool2d":
            return torch.nn.Sequential( *layers[::-1] )
        assert False, f"ResNet3D.create_multi_layer: Unknown multiple layer: {name}"
            
    def layer_to_3D(self, layer):
        if isinstance(layer, torch.nn.Conv2d):
            return torch.nn.Conv3d(in_channels      = layer.in_channels,
                                    out_channels    = layer.out_channels,
                                    kernel_size     = layer.kernel_size[0],
                                    stride          = layer.stride[0],
                                    padding         = layer.padding[0],
                                    bias            = layer.bias)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            if self.normalization == "batch":
                return torch.nn.BatchNorm3d(num_features        = layer.num_features,
                                            eps                 = layer.eps,
                                            momentum            = layer.momentum,
                                            affine              = layer.affine,
                                            track_running_stats = layer.track_running_stats)
            elif self.normalization == "layer":
                return LayerNorm()
            elif self.normalization == "group":
                if layer.num_features % 16 == 0:
                    return torch.nn.GroupNorm(16, layer.num_features)
                return torch.nn.GroupNorm(8, layer.num_features)
        elif isinstance(layer, torch.nn.ReLU):
            return layer
        elif isinstance(layer, torch.nn.MaxPool2d):
            return torch.nn.MaxPool3d(kernel_size   = layer.kernel_size,
                                    stride          = layer.stride,
                                    padding         = layer.padding,
                                    dilation        = layer.dilation,
                                    ceil_mode       = layer.ceil_mode)
        elif isinstance(layer, torch.nn.Flatten):
            return layer
        elif isinstance(layer, torch.nn.AdaptiveAvgPool2d):
            return torch.nn.AdaptiveAvgPool3d(output_size = layer.output_size)
        elif isinstance(layer, torch.nn.Linear):
            return layer
        elif isinstance(layer, torch.nn.Identity):
            return layer
        elif isinstance(layer, torch.nn.AvgPool2d):
            return torch.nn.AvgPool3d(kernel_size   = 3,
                                    stride          = 2,
                                    padding         = 1)
        elif isinstance(layer, timm.models.layers.drop.DropBlock2d):
            return drop_block.DropBlock3d(p = layer.drop_prob, block_size = layer.block_size)
        assert False, f"ResNet3D.layer_to_3D: Unknown layer {layer}"
    
    def to_dict(self):
        return {"version": self.version, 
                "n_features": self.n_features,
                "drop_rate": self.drop_rate,
                "drop_block_rate": self.drop_block_rate,
                "normalization": self.normalization,
                "remove_first_maxpool": self.remove_first_maxpool}

    
class ResidualBlock3D(torch.nn.Module):
    def __init__(self, layers):
        super(ResidualBlock3D, self).__init__()
        if isinstance(layers[-1], drop_block.DropBlock3d):
            self.dropblock  = layers[-1]
            layers          = layers[:-1]
        else:
            self.dropblock = None
        n_layers = len(layers)
        if (n_layers == 7) or (n_layers == 10): 
            # 7 and 10 layers for Basic and Bottleneck blocks, respectively
            self.downsample = layers[-1] 
            self.activation = layers[-2]
            layers = layers[:-2]
        elif (n_layers == 6) or (n_layers == 9):
            # 6 and 9 layers for Basic and Bottleneck blocks, respectively
            self.downsample = None
            self.activation = layers[-1]
            layers = layers[:-1]
        else:
            assert False, f"ResidualBlock3D.__init__: invalid number of layers {n_layers} for residual block"
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.activation(x)
        if self.dropblock is not None:
            x = self.dropblock(x)
        return x


if __name__ == "__main__":
    r = ResNet3D(version = "resnet34", drop_block_rate = 0.1, drop_rate = .8, normalization = "group")
    # , normalization = "layer", drop_block_rate = 0.1)
    # print(r)
    # sample = torch.rand(2,1,45*2,180//3,91)
    # sample = torch.rand(2,1,45*2,180,91)
    # r(sample)
    # from torchsummary import summary
    # summary(r, (1, 45, 180, 91))
    
