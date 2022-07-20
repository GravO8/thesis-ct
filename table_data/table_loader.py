import sys
sys.path.append("..")
from utils.csv_loader import CSVLoader

class TableLoader:
    def preprocess(self, ampute: bool = False, impute: bool = False):
        self.filter_no_ncct()
        self.add_vars()
        if impute: 
            assert not ampute
            self.impute()
        if ampute: 
            assert not impute
            self.ampute()
        
    def filter_no_ncct(self):
        self.table = self.table[self.table["NCCT"] == "OK"]
        
        
    
