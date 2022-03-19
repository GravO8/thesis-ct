import timm, torch


class LayerNorm(torch.nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.norm = None
    def forward(self, x):
        if self.norm is None:
            _, C, Z, H, W   = x.shape
            self.norm       = torch.nn.LayerNorm([C, Z, H, W], elementwise_affine = False)
        return self.norm(x)


class ResNet3D(torch.nn.Module):
    def __init__(self, version: str = "resnet18", in_channels = 1, n_features = "same",
    dropout: float = None, normalization = "batch"):
        super(ResNet3D, self).__init__()
        assert "resnet" in version
        assert normalization in ("batch", "layer")
        self.version        = version
        self.in_channels    = in_channels
        self.n_features     = n_features
        self.dropout        = dropout
        self.normalization  = normalization
        self.layers         = torch.nn.ModuleList( self.to_layer_list() )
        if n_features == "same":
            self.layers[-1] = torch.nn.Identity()
        elif isinstance(self.n_features, int) or self.n_features.isnumeric():
            self.layers[-1] = torch.nn.LazyLinear(int(self.n_features))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def to_layer_list(self):
        layers      = {}
        layers_len  = {}
        depth       = 0
        first       = True
        for layer in timm.create_model(self.version, in_chans = self.in_channels).modules():
            if first:
                layers_len[0]   = len(layer._modules)
                layers[0]       = []
                first           = False
            elif len(layer._modules) > 0:
                depth              += 1
                layers[depth]       = [layer.__class__.__name__]
                layers_len[depth]   = len(layer._modules)
            else:
                layers[depth].append( self.layer_to_3D(layer) )
                layers_len[depth] -= 1
            while (layers_len[depth] == 0) and (depth != 0):
                layers[depth-1].append( self.create_multi_layer(layers[depth]) )
                layers_len[depth-1] -= 1
                layers[depth]        = []
                depth               -= 1
        return layers[0]

    def create_multi_layer(self, layers):
        name, layers = layers[0], layers[1:]
        if name == "Sequential":
            return torch.nn.Sequential( *layers )
        elif (name == "BasicBlock") or (name == "Bottleneck"):
            block = ResidualBlock3D( layers )
            if self.dropout is None:
                return block
            else:
                return torch.nn.Sequential(block, torch.nn.Dropout(p = self.dropout))
        elif name == "SelectAdaptivePool2d":
            return torch.nn.Sequential( *layers[::-1] )
        else:
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
        else:
            assert False, f"ResNet3D.layer_to_3D: Unknown layer {layer}"

    def print_layer_list(self, l, d = 0):
        for e in l:
            if isinstance(e, list):
                print(" "*d, end = "")
                print(f"{e[0]}({len(e[1:])})")
                self.print_layer_list(e[1:], d + 4)
            else:
                print(" "*d, end = "")
                print(e)
    
    def to_dict(self):
        return {"version": self.version, "n_features": self.n_features, 
            "dropout": "no" if self.dropout is None else self.dropout,
            "normalization": self.normalization}

    
class ResidualBlock3D(torch.nn.Module):
    def __init__(self, layers):
        super(ResidualBlock3D, self).__init__()
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
            assert False, f"ResidualBlock3D.__init__:invalid number of layers {n_layers} for residual block"
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        shortcut = x
        for layer in self.layers[:-1]:
            x = layer(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        return self.activation(x)


if __name__ == "__main__":
    r = ResNet3D(version = "resnet34", normalization = "layer")
    print(r)
    sample = torch.rand(2,1,45,180,91)
    r(sample)
    # print( timm.create_model("resnet50d") )
