import sys
sys.path.append("..")
from models.models import get_timm_model
from models.model import Axial2DCNN
from utils.main import main

if __name__ == "__main__":
    to_test = [ Axial2DCNN(get_timm_model("resnet18", global_pool = "gap")),
                Axial2DCNN(get_timm_model("resnet18", global_pool = "gap", pretrained = True, frozen = True))]
    device  = 1
    heights = ["B", "C"] 
    for height in heights:
        main(to_test, device = device, slice = height, slice_range = 1)
    
    
