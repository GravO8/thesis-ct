import torch
# https://iksinc.online/2020/05/10/groups-parameter-of-the-convolution-layer/

'''
x1 = [a1, b1, c1]
x2 = [a2, b2, c2]
x  = [a1, a2, b1, b2, c1, c2] (tangle)
x  = [conv(a1,a2), conv(b1,b2), conv(c1,c2)] group convolution
'''

def tangle(x1, x2):
    assert x1.shape == x2.shape
    shp     = list(x1.shape)
    shp[1] *= 2
    x       = torch.zeros(shp)
    x[:,[i for i in range(shp[1]) if i%2 == 0],:,:,:] = x1
    x[:,[i for i in range(shp[1]) if i%2 == 1],:,:,:] = x2
    return x
    

if __name__ == '__main__':
    x0 = torch.zeros(32,64,5,5,5)
    x1 = torch.zeros(32,64,5,5,5)+1
    x = tangle(x0, x1)
    
    conv = torch.nn.Conv3d(in_channels = 64*2, out_channels = 64, groups = 64,
    kernel_size = 5, padding = 2)
    x = conv(x)
    print(x.shape)
    print(conv.weight.shape)
    print(conv.bias.shape)
