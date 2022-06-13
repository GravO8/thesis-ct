from torchsummary import summary
from .encoder import Encoder
from .models import conv
from .model import Model
import torch
import numpy as np


def kaiming_init(m):
    # copied from https://github.com/1Konny/Beta-VAE/blob/977a1ece88e190dd8a556b4e2efb665f759d0772/model.py#L148
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def deconv(dim, *args, **kwargs):
    deconv_4 = torch.nn.ConvTranspose2d if dim == 2 else torch.nn.ConvTranspose3d
    return torch.nn.Sequential(
        torch.nn.ReLU(inplace = True),
        deconv_4(*args, **kwargs))

def vae(name, in_channels = 1, n_start_chans = 8, dim = 3, N = 6, shape = (46, 109, 91), 
    bias = False, init_blocks = None, add_sigmoid = False):
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
    conv_4   = lambda in_, out_: conv(dim, in_, out_, 4, 2, 1, bias = bias)
    deconv_4 = lambda in_, out_: deconv(dim, in_, out_, 4, 2, 1, bias = bias)
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
    if add_sigmoid:
        decoder.append( torch.nn.Sigmoid() )
    return VAEModel(name, torch.nn.Sequential( *encoder ), torch.nn.Sequential( *decoder ), chans*2, init_blocks = init_blocks)
    
def vae_v1():
    return vae("vae_1", in_channels = 1, dim = 3, n_start_chans = 8, N = 6, shape = (64, 128, 128), bias = False, add_sigmoid = False)
    
def vae_v2():
    return vae("vae_v2", in_channels = 1, dim = 3, n_start_chans = 8, N = 6, shape = (64, 128, 128), bias = True, init_blocks = kaiming_init, add_sigmoid = True)
    
def vae_v3():
    return vae("vae_v3", in_channels = 1, dim = 3, n_start_chans = 16, N = 6, shape = (64, 128, 128), bias = True, init_blocks = kaiming_init, add_sigmoid = True)
    
    
def reparametrize(mu, logvar):
    # copied from https://github.com/1Konny/Beta-VAE/blob/977a1ece88e190dd8a556b4e2efb665f759d0772/model.py#L10
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std*eps
    
    
class Squeeze(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
    def forward(self, x):
        x = x.reshape((x.shape[0], self.z_dim))
        return x


class VAEModel(torch.nn.Module):
    def __init__(self, name, encoder, decoder, z_dim, init_blocks = None):
        super().__init__()
        self.name    = name
        self.encoder = torch.nn.Sequential(encoder, Squeeze(z_dim), torch.nn.Linear(z_dim, z_dim*2))
        self.decoder = decoder
        self.z_dim   = z_dim # number of features of the latent space
        if init_blocks is not None:
            for block in self._modules:
                for m in self._modules[block]:
                    init_blocks(m)

    def forward(self, x):
        x             = x / 100
        distributions = self.encoder(x)
        mu            = distributions[:, :self.z_dim]
        logvar        = distributions[:, self.z_dim:]
        z             = reparametrize(mu, logvar)
        z             = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # adds the x, y, z dimensions
        x_recon       = self.decoder(z)
        return x_recon, mu, logvar

    def get_name(self):
        return self.name
