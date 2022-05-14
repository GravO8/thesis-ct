import torch

def final_mlp(in_features):
    return torch.nn.Sequential(torch.nn.Linear(in_features,1), torch.nn.Sigmoid())
    
def conv_3d(in_channels, out_channels, kernel_size, stride = 1, padding = None):
    return torch.nn.Sequential( torch.nn.Conv3d(in_channels     = in_channels,
                                                out_channels    = out_channels,
                                                kernel_size     = kernel_size,
                                                stride          = stride,
                                                padding         = (kernel_size-1)//2 if padding is None else padding,
                                                bias            = True),
                                torch.nn.BatchNorm3d(num_features = out_channels),
                                torch.nn.ReLU(inplace = True))

def custom_3D_cnn_v1():
    return torch.nn.Sequential( 
        conv_3d(1,8,5), 
        conv_3d(8,16,3,2,1), 
        conv_3d(16,32,3), 
        conv_3d(32,64,3,2,1),
        torch.mean(2,3,4) ) # Global Average Pooling
