import torch, numpy
from .encoder import Encoder
from .model import Model, init_xavier_normal, init_kaiming_normal
from abc import ABC, abstractmethod


class MILPooling(ABC):
    @abstractmethod
    def __call__(self, x):
        pass
    @abstractmethod
    def get_name(self):
        pass
        
class MILPoolingEncodings(MILPooling):
    def __call__(self, x):
        assert x.dim() == 2, f"MILPoolingEncodings.__call__: expected tensor of size 2 (number of instances, instance encoding size) but got {x.dim()}"
        return self.forward(x)
    @abstractmethod
    def forward(self, x):
        pass
    def predict_attention(self, x):
        assert False, "MILPoolingEncodings.predict_attention: not implemented"
    
class MaxMILPooling(MILPoolingEncodings):
    def forward(self, x):
        return x.max(dim = 0).values
    def get_name(self):
        return "MaxPooling"
    def predict_attention(self, x):
        return self.forward(x), x.argmax(dim = 0)
        
class MeanMILPooling(MILPoolingEncodings):
    def forward(self, x):
        return x.mean(dim = 0)
    def get_name(self):
        return "MeanPooling"
        
class AttentionMILPooling(MILPoolingEncodings, torch.nn.Module):
    def __init__(self, in_channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attention = torch.nn.Sequential(
                            torch.nn.Linear(in_channels, bottleneck_dim),
                            torch.nn.Tanh(),
                            torch.nn.Linear(bottleneck_dim, 1))
    def attention_fn(self, x):
        a = self.attention(x)
        a = torch.nn.Softmax(dim = 0)(a).T
        return a
    def forward(self, x):
        a = self.attention_fn(x)
        x = torch.mm(a,x)
        return x
    def get_name(self):
        return "AttentionPooling"
    def predict_attention(self, x):
        return self.forward(x), self.attention_fn(x)


class MILEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, mil_pooling: MILPooling, feature_extractor: Encoder = None):
        super().__init__()
        self.encoder           = encoder # instance encoder
        self.mil_pooling       = mil_pooling
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            self.out_channels  = self.encoder.out_channels
        else:
            self.out_channels  = self.feature_extractor.out_channels
    def get_name(self):
        return f"{self.encoder.get_name()}-{self.mil_pooling.get_name()}" + ("" if self.feature_extractor is None else "_"+self.feature_extractor.get_name())
    def forward(self, x):
        x = self.encoder(x)
        x = self.mil_pooling(x)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return x
    def predict_attention(self, x):
        x    = self.encoder(x)
        x, a = self.mil_pooling.predict_attention(x)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return x, a
    def init_weights(self, init_convs: bool):
        init_kaiming_normal(self.encoder, init_convs = init_convs)
        # if isinstance(self.mil_pooling, AttentionMILPooling):
            # self.mil_pooling.load_state_dict(torch.load("attention_pooling_pretrained.pt"))
        # else:
        init_kaiming_normal(self.mil_pooling, init_convs = init_convs)
        if self.feature_extractor is not None:
            # init_kaiming_normal(self.feature_extractor, init_convs = init_convs)
            self.feature_extractor.load_state_dict(torch.load("mean_pooling_feature_extractor.pt"))
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        
class MILNet(Model):
    def __init__(self, encoder: MILEncoder):
        assert isinstance(encoder, MILEncoder), "MILNet.__init__: 'encoder' must be of class 'MILEncoder'"
        super().__init__(encoder)
    def forward(self, batch):
        out = []
        for scan in batch.unbind(dim = 0):
            out.append( super().forward(scan) )
        return torch.stack(out, dim = 0)
    def name_appendix(self):
        return "MILNet"
    def predict_attention(self, x):
        x, a = self.encoder.predict_attention(x)
        return self.mlp(x), a
    def init_weights(self, init_convs: bool):
        self.encoder.init_weights(init_convs = init_convs)
        # init_xavier_normal(self.mlp)
        self.mlp.load_state_dict(torch.load("mean_pooling_mlp.pt"))
        for param in self.mlp.parameters():
            param.requires_grad = False
        
class MILNetAfter(MILNet):
    def __init__(self, encoder: MILEncoder):
        super().__init__(encoder)
        assert encoder.encoder.global_pool is not None
        assert (encoder.feature_extractor is None) or (encoder.feature_extractor.global_pool is None)
    def name_appendix(self):
        return super().name_appendix() + "-after"


class MILAfterAxial(MILNetAfter):
    def process_input(self, x):
        x = self.normalize_input(x)
        x = x.permute((3,0,1,2)) # (C,x,y,z) = (1,x,y,z) -> (z,C,x,y) = (B,1,x,y)
        # x = x[[i for i in range(x.shape[0]) if torch.count_nonzero(x[i,:,:,:] > 0) > 100]]
        return x
    def name_appendix(self):
        return super().name_appendix() + "-Axial"


class MILTensorAxial(MILNetAfter):
    def process_input(self, x):
        return x
    def name_appendix(self):
        return super().name_appendix() + "-TensorAxial"


def to_blocks(arr_in: torch.Tensor, block_shape: tuple):
    '''
    code adapted from
    https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py
    to work with torch tensors
    '''
    block_shape = numpy.array(block_shape)
    arr_shape   = numpy.array(arr_in.shape)
    new_shape   = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.stride() * block_shape) + arr_in.stride()
    arr_out     = torch.as_strided(arr_in, size=new_shape, stride=new_strides)
    return arr_out
    
class MILAfterBlock(MILNetAfter):
    def process_input(self, x, debug = False):
        x = self.normalize_input(x)
        x = x[:,:-1,:-1,:-1].squeeze()      # trim input from (91,109,91) to (90,108,90) to be more evenly divisible
        x = to_blocks(x, (18,27,18))
        x = x.reshape((100, 1, 18, 27, 18)) # (B,C,x,y,z)
        if debug:
            import matplotlib.pyplot as plt
            for slice in range(0,x.shape[0],20):
                for z in range(x.shape[-1]):
                    _, axs = plt.subplots(5, 4)
                    for i in range(5):
                        for j in range(4):
                            axs[i][j].imshow(x[slice+i*4+j,:,:,:,z].squeeze(), cmap = "gray")
                    plt.show()
        return x
    def name_appendix(self):
        return super().name_appendix() + "-Block"
