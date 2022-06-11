from torchsummary import summary
from .encoder import Encoder
from .models import conv
from .model import Model
import torch
import numpy as np

'''
(46, 109, 91)
1, 2
(23, 55, 46)
2, 4
(12, 28, 23)
3, 8
(6, 14, 12)
4, 16
(3, 7, 6)
5, 32
(2, 4, 3)
6, 64
(1, 1, 1)
128
'''

def deconv(dim, *args, **kwargs):
    deconv_4 = torch.nn.ConvTranspose2d if dim == 2 else torch.nn.ConvTranspose3d
    return torch.nn.Sequential(
        torch.nn.ReLU(inplace = True),
        deconv_4(*args, **kwargs))

def vae_v1(in_channels = 1, n_start_chans = 8, dim = 3, N = 6, shape = (46, 109, 91)):
    '''
    1   (64, 128, 128)
    8   (32, 64,  64)
    16  (16, 32,  32)
    32  (8,  16,  16)
    64  (4,  8,   8)
    128 (2,  4,   4)
    256 (1,  1,   1)
    '''
    shape    = (np.array(shape)/2).astype(int)
    conv_4   = lambda in_, out_: conv(dim, in_, out_, 4, 2, 1, bias = False)
    deconv_4 = lambda in_, out_: deconv(dim, in_, out_, 4, 2, 1, bias = False)
    chans    = n_start_chans
    encoder  = [conv_4(in_channels, chans)]
    decoder  = [deconv_4(chans, in_channels)]
    for i in range(N-2):
        encoder.append( conv_4(chans, chans*2) )
        decoder = [deconv_4(chans*2, chans)] + decoder
        chans *= 2
        shape  = (.5*shape).astype(int)
    encoder.append( conv(dim, chans, chans*2, kernel_size = shape, stride = 1, padding = 0) )
    decoder = [deconv(dim, chans*2, chans, kernel_size = shape, stride = 1, padding = 0)] + decoder
    return torch.nn.Sequential( *encoder ), torch.nn.Sequential( *decoder )


# class VAEEncoder(torch.nn.Module):
#     def __init__(self):
#         '''
#         VAE encoders are just going to be sequences of 3D convolutions 
#         'layers' is a a list of integers with the number of channels in each layer
#         of these 3D conv layers 
#         '''
#         super().__init__()
#         assert dim in (2,3)
#         self.dim = dim
#         self.init_encoder(layers, downsample)
#         self.mlp = torch.nn.Linear(self.n_features(), 2*self.n_features())
#     def init_encoder(self, layers, downsample):
#         conv = conv_2d if self.dim == 2 else conv_3d
#         layers = [conv(1,layers[0],)]
#         for i in range(1,len(layers)):
# 
#     def n_features(self):
#         return self.encoder.out_channels
#     def normalize_input(self, x, range_max: int = 100):
#         return x/range_max
#     def name_appendix(self):
#         return "BetaVAE"
#     def get_deconv_decoder(self, x_shape = (1, 46, 109, 91)):
#         x = torch.randn(x_shape)
#         shapes = [x_shape]
# 
#         for layer in self.encoder.children():
#             print(layer)
#         # summary(self.encoder, x_shape)
# 
# 
# class VAEModel:
#     def __init__(self, encoder: VAEEncoder, decoder):
#         self.encoder = encoder
#         self.decoder = decoder
#         self.z_dim   = self.encoder.n_features() # number of features of the latent space
# 
#     def forward(self, x):
#         distributions = self.encoder(x)
#         mu            = distributions[:, :self.z_dim]
#         logvar        = distributions[:, self.z_dim:]
#         z             = reparametrize(mu, logvar)
#         x_recon       = self.decoder(z)
#         return x_recon, mu, logvar
# 
#     def get_name(self):
#         return self.encoder.get_name() + "-" + self.decoder.get_name()
