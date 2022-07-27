import sys
sys.path.append("..")
from utils.csv_loader import CSVLoader


class RadiomicsLoader(CSVLoader):
    def __init__(self, csv_filename: str, keep_cols: list, target: str, 
    binary: bool = True, **kwargs):
        assert target in ("aspects", "rankin")
        target_col = f"binary_{target}" if binary else target
        set_col    = f"{target}_set"
        super().__init__(csv_filename, keep_cols, target_col, 
            set_col             = set_col,
            empty_values_method = None, 
            **kwargs)
    
    def preprocess(self, keep_cols = "all"):
        self.table = self.table[self.table["Mask"] == "MNI152_T1_2mm_brain_mask"].copy()
        if keep_cols == "all":
            return self.all_cols()
        self.assert_cols(keep_cols)
        return keep_cols
        
    def assert_cols(self, keep_cols):
        forbidden_cols = ["Image", "Mask", "rankin", "binary_rankin", "rankin_set", 
        "M1", "M2", "M3", "M4", "M5", "M6", "C", "IC", "I", "L", "aspects", 
        "binary_aspects", "aspects_set"]
        for col in forbidden_cols:
            assert col not in keep_cols
            
    def all_cols(self):
        return [col for col in self.table.columns if col.startswith("original_")]
