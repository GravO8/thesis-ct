import sys
sys.path.append("..")
from utils.csv_loader import CSVLoader


class RadiomicsLoader(CSVLoader):
    def preprocess(self):
        
