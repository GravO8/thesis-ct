import sys
sys.path.append("..")
from models.models import get_timm_model
from models.model import Axial2DCNN
from utils.main import main

if __name__ == "__main__":
    to_test = [ Axial2DCNN(get_timm_model("resnet18", global_pool = "gap", pretrained = True, frozen = False)),
                Axial2DCNN(get_timm_model("resnet18", global_pool = "gap", pretrained = True, frozen = True)),
                Axial2DCNN(get_timm_model("resnet34", global_pool = "gap")),
                Axial2DCNN(get_timm_model("resnet50", global_pool = "gap")),
                Axial2DCNN(custom_2D_cnn_v1(global_pool = "gap"))]
    device  = ???
    for slice_range in slice_ranges:
        main(to_test, device = device, slice = "A", slice_range = ???)
    
    
