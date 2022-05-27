import sys
sys.path.append("..")
from models.models import get_timm_model, custom_2D_cnn_v1
from models.model import Axial2DCNN
from utils.main import main

if __name__ == "__main__":
    to_test = [ Axial2DCNN(get_timm_model("efficientnet_b0", global_pool = "gap")),
                Axial2DCNN(get_timm_model("efficientnet_b0", global_pool = "gap", pretrained = True, frozen = True)),
                Axial2DCNN(get_timm_model("efficientnet_b1", global_pool = "gap")),
                Axial2DCNN(get_timm_model("efficientnet_b1", global_pool = "gap", pretrained = True, frozen = True))]
    device      = 3
    slice_range = 2
    main(to_test, device = device, slice = "A", slice_range = slice_range)
