import sys, torch
sys.path.append("..")
from utils.half_ct_loader import HalfCTLoader
from utils.main import set_home
from models.vae import vae_v1
from torchsummary import summary
# from models.models import custom_3D_cnn_v1

if __name__ == "__main__":
    dir = set_home(0)
    loader = HalfCTLoader(data_dir = dir, pad = (64, 128, 128))
    _, _, test_set = loader.load_dataset()
    
    patient = test_set[0]["ct"]["data"]
    affine = test_set[0]["ct"].affine
    
    # encoder, decoder = vae_v1(N = 6, shape = (64, 128, 128))
    # x = torch.randn([1,64,128,128])
    # s = torch.randn([256,1,1,1])
    # # summary(encoder, x.shape)
    # summary(decoder, s.shape)
