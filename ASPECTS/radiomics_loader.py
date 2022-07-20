import sys
sys.path.append("..")
from utils.csv_loader import CSVLoader


class RadiomicsLoader(CSVLoader):
    def preprocess(self):
        self.table = self.table[self.table["Mask"] == "MNI152_T1_2mm_brain_mask"].copy()
