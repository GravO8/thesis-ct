import sys
sys.path.append("..")
from utils.table_classifier import TableClassifier
from table_loader import TableLoader

class ASTRALClassifier(TableClassifier):
    def __init__(self, dataset_filename: str, **kwargs):
        loader = TableLoader(csv_filename       = dataset_filename, 
                            keep_cols           = ["age", "totalNIHSS-5", "time_since_onset", "altVis-5", "altCons-5", "gliceAd-4"],
                            target_col          = "binary_rankin",
                            normalize           = False,
                            **kwargs)
        super().__init__(loader)
        self.format_cols()
        
    def format_cols(self):
        for s in self.loader.available_sets():
            self.loader.set_col(s, "age",                   (self.loader.get_col(s, "age") // 5).astype(int))
            self.loader.set_col(s, "totalNIHSS-5",           self.loader.get_col(s, "totalNIHSS-5").astype(int))
            self.loader.set_col(s, "time_since_onset",   (2*(self.loader.get_col(s, "time_since_onset") >= 3)).astype(int))
            self.loader.set_col(s, "altVis-5",           (2*(self.loader.get_col(s, "altVis-5") > 0)).astype(int))
            self.loader.set_col(s, "altCons-5",          (3*(self.loader.get_col(s, "altCons-5") > 0)).astype(int))
            glucose = self.loader.get_col(s, "gliceAd-4")
            self.loader.set_col(s, "gliceAd-4",           ((glucose > 131) | (glucose < 66)).astype(int))
        
    def predict(self, x):
        x = x.sum(axis = 1)
        x = x > 30
        return x.astype(int)
        

if __name__ == "__main__":
    astral = ASTRALClassifier("table_data.csv",
                              dirname             = "../../../data/gravo",
                              set_col             = "set",
                              join_train_val      = True,
                              join_train_test     = False,
                              filter_out_no_ncct  = True,
                              reshuffle           = False,
                              # empty_values_method = "impute")
                              empty_values_method = "amputate")
    print(astral.compute_metrics("train"))
