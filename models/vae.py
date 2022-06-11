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
    return torch.nn.Sequential( *encoder ), torch.nn.Sequential( *decoder ), chans*2

class VAEModel:
    def __init__(self, name, encoder, decoder, z_dim):
        self.name    = name
        self.encoder = torch.nn.Sequential(encoder, torch.nn.Linear(z_dim, z_dim*2))
        self.decoder = decoder
        self.z_dim   = self.z_dim # number of features of the latent space

    def forward(self, x):
        distributions = self.encoder(x)
        mu            = distributions[:, :self.z_dim]
        logvar        = distributions[:, self.z_dim:]
        z             = reparametrize(mu, logvar)
        x_recon       = self.decoder(z)
        return x_recon, mu, logvar

    def get_name(self):
        return name
