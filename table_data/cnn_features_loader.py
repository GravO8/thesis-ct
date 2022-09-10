import sys, numpy as np
sys.path.append("..")
from table_data.stages import *
from utils.csv_loader import CSVLoader
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
    

class CNNFeaturesLoader(CSVLoader):
    def preprocess(self, keep_cols):
        if keep_cols == "all":
            return [col for col in list(self.table.columns) if col.startswith("feature")]
        return keep_cols
                
    def remove_outliers(self):
        pass
        
    def feature_selection(self):
        # sel = SelectKBest(mutual_info_classif, k = 128)
        # sel.fit(self.sets["train"]["x"], self.sets["train"]["y"])
        # for set in self.available_sets():
        #     self.sets[set]["x"] = self.sets[set]["x"][self.sets[set]["x"].columns[sel.get_support(indices = True)]]
        pca = PCA(n_components = 128)
        pca.fit(self.sets["train"]["x"])
        for set in self.available_sets():
            self.sets[set]["x"] = pca.transform(self.sets[set]["x"])
        

if __name__ == "__main__":
    pass
