import sys
sys.path.append("../table_data")
from classic_classifier import knns
from radiomics_loader import RadiomicsLoader

loader = RadiomicsLoader(table_filename = "ncct_radiomic_features.csv", 
                        data_dir = "../../data/gravo",
                        target = "binary_rankin")
                        
knns(loader)
