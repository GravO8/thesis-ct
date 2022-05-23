import sys
sys.path.append("..")
from models.models import get_timm_model
from models.model import Axial2DCNN
from utils.main import main

if __name__ == "__main__":
    to_test = [ Axial2DCNN(get_timm_model("resnet18", global_pool = "gap")) ]
    device  = 1
    slice_ranges = [1,3,5,7,9]
    for slice_range in slice_ranges:
        main(to_test, device = device, slice = "A", slice_range = slice_range)
    
    
