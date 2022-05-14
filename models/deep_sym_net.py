import torch

class InceptionModule3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels: int = 64):
        torch.nn.Module.__init__(self)
        self.path1 = torch.nn.Conv3d(in_channels  = in_channels, 
                                     out_channels = out_channels,
                                     kernel_size  = 1)
        self.path2 = torch.nn.Sequential(
                        torch.nn.Conv3d(in_channels  = in_channels,
                                        out_channels = out_channels,
                                        kernel_size  = 1),
                        torch.nn.Conv3d(in_channels  = out_channels,
                                        out_channels = out_channels,
                                        padding      = 1,
                                        kernel_size  = 3))
        self.path3 = torch.nn.Sequential(
                        torch.nn.Conv3d(in_channels  = in_channels,
                                        out_channels = out_channels,
                                        kernel_size  = 1),
                        torch.nn.Conv3d(in_channels  = out_channels,
                                        out_channels = out_channels,
                                        padding      = 2,
                                        kernel_size  = 5))
        self.path4 = torch.nn.Sequential(
                        torch.nn.MaxPool3d(kernel_size  = 3, 
                                           stride       = 1,
                                           padding      = 1),
                        torch.nn.Conv3d(in_channels  = in_channels,
                                        out_channels = out_channels,
                                        kernel_size  = 1))

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        return torch.cat([x1, x2, x3, x4], dim = 1)
        
if __name__ == "__main__":
    im = InceptionModule3D(1)
    x = torch.randn(32,1,91,109,91)
    print(im(x).shape)
        
