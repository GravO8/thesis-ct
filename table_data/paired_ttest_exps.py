import os, numpy as np
from table_loader import TableLoader
from utils.classic_classifiers import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from clinic_classifiers import ASTRALClassifier

TWO_VARS        = ["age", "totalNIHSS-5"]
ASTRAL          = TWO_VARS + ["time_since_onset", "altVis-5", "altCons-5", "gliceAd-4"]

LEUKOARAIOSIS   = ASTRAL + ["leucoa-7"]
ASPECTS         = ASTRAL + ["aspects-7"]

OCCLUSION       = ASTRAL + ["ocEst-10"]
OCC_ASPECTS     = OCCLUSION + ["aspects-7"]
ALL_BRAIN       = OCC_ASPECTS + ["leucoa-7"]

OCCLUSION_PRED  = ASTRAL + ["occlusion-pred2"]
OCC2_ASPECTS    = OCCLUSION_PRED + ["aspects-7"]
ALL_BRAIN2      = OCCLUSION_PRED + ["aspects-7"]

ALL             = ALL_BRAIN + ["occlusion-pred2"]


DIR     = "../../../data/gravo"
DATASET = "table_data.csv"
MISSING = "amputate"
N_ITER  = 40
CV      = 5
METRIC  = "f1"
EXPS    = { "2vars":                                TWO_VARS,
            "ASTRAL_vars":                          ASTRAL,
            
            "leukoaraiosis":                        LEUKOARAIOSIS,
            "aspects":                              ASPECTS,
            
            "occlusion":                            OCCLUSION,
            "occlusion+aspects":                    OCC_ASPECTS,
            "occlusion+aspects+leukoaraiosis":      ALL_BRAIN,
            
            "occlusion_pred":                       OCCLUSION_PRED,
            "occlusion_pred+aspects":               OCC2_ASPECTS,
            "occlusion_pred+aspects+leukoaraiosis": ALL_BRAIN2
            }


class DummyLoader:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.sets = {"train": {"x": x_train, "y": y_train}, 
                    "test":   {"x": x_test, "y": y_test}}
    def available_sets(self):
        return [s for s in self.sets]
    def get_set(self, set):
        return self.sets[set]
    def set_col(self, set: str, col: str, values):
        self.sets[set]["x"][col] = values
    def get_col(self, set: str, col: str):
        return self.sets[set]["x"][col].values
        
def normalize(train_x, test_x):
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x), columns = train_x.columns)
    test_x  = pd.DataFrame(scaler.transform(test_x),  columns = test_x.columns)
    return train_x, test_x

loader = TableLoader(DATASET,
                    keep_cols           = ALL,
                    target_col          = "binary_rankin",
                    normalize           = False,
                    dirname             = DIR,
                    join_train_val      = True,
                    join_train_test     = True,
                    reshuffle           = True,
                    set_col             = "all",
                    filter_out_no_ncct  = False,
                    empty_values_method = "amputate")
set  = loader.get_set("train")
x, y = set["x"], np.array(set["y"])
fold = 0
for train_index, test_index in StratifiedKFold(n_splits = 10, shuffle = False).split(x, y):
    fold += 1
    print("fold", fold)
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_train_norm, x_test_norm = normalize(x_train, x_test)
    for exp in EXPS:
        cols = EXPS[exp]
        name = f"ttest-{exp}"
        x_train_exp, x_test_exp = x_train_norm[cols], x_test_norm[cols]
        print("   ", name, x_train_exp.shape, x_test_exp.shape)
        loader     = DummyLoader(x_train_exp, x_test_exp, y_train, y_test)
        classifier = logistic_regression(loader, n_iter = N_ITER, metric = METRIC, cv = CV)
        classifier.record_performance(f"{name}-{fold}", MISSING, run_name = f"runs-{name}")
    name = "ttest-ASTRAL"
    x_train_exp, x_test_exp = x_train[ASTRAL].copy(), x_test[ASTRAL].copy()
    print("   ", name, x_train_exp.shape, x_test_exp.shape)
    loader = DummyLoader(x_train_exp, x_test_exp, y_train, y_test)
    astral = ASTRALClassifier(dataset_filename = None, loader = loader)
    astral.record_performance(MISSING, run_name = f"runs-{name}")
    print()
    print()
