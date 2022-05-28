import sys
sys.path.append("..")
from utils.ensemble import majority_ensemble, average_ensemble, weighted_average_ensemble

if __name__ == "__main__":
    experiments = ["axial-range/Axial1-2DCNN-resnet18_gap_2D", 
                   "axial-height/Axial1B-2DCNN-resnet18_gap_2D",
                   "axial-height/Axial1C-2DCNN-resnet18_gap_2D"]
    experiments_dir = "../../../runs/systematic"
    data_dir = "../../../data/gravo/"
    ensemble = weighted_average_ensemble(experiments, 
                                experiments_dir = experiments_dir,
                                data_dir = data_dir)
    ensemble.record_performance()
    
