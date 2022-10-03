import os, numpy as np
from table_loader import TableLoader
from utils.classic_classifiers import *
from sklearn.model_selection import StratifiedKFold
from clinic_classifiers import ASTRALClassifier


DIR = "../../../data/gravo"
DATASET = "table_data.csv"
N_ITER = 50
CV     = 5
METRIC = "f1"
EXPS   = {"2vars": ["age","totalNIHSS-5"],
          "occlusion": ["altura-1", "peso-1", "age", "hemoAd-4", "hemat-4",
                        "inrAd-4", "gliceAd-4", "totalNIHSS-5", "preArtSis-5", "preArtDia-5", 
                        "ocEst-10"],
          "occlusion_pred": ["altura-1", "peso-1", "age", "hemoAd-4", "hemat-4",
                        "inrAd-4", "gliceAd-4", "totalNIHSS-5", "preArtSis-5", "preArtDia-5", 
                        "occlusion-pred2"]
        }
ASTRAL_COLS = ["age", "totalNIHSS-5", "time_since_onset", "altVis-5", "altCons-5", "gliceAd-4"]


NEW_COL         = "occlusion-pred2"
table_data      = pd.read_csv(os.path.join(DIR,DATASET))
occlusion_pred  = pd.read_csv("dataset-occlusion.csv")
preds           = []
for _, row in table_data.iterrows():
    pred = occlusion_pred[occlusion_pred.patient_id.astype(str) == row.idProcessoLocal]["occlusion-pred"].values
    if len(pred) > 0:
        preds.append(int(pred[0] > .5))
    else:
        gt = row["ocEst-10"]
        if gt != "None":
            preds.append(gt)
        else:
            preds.append("")
table_data[NEW_COL] = preds
table_data.to_csv(os.path.join(DIR,DATASET), index = False)


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


for missing in ("amputate", "impute", "impute_mean"):
    # stage,missing_values,model,set,f1_score,accuracy,precision,recall,auc
    loader  = TableLoader(DATASET,
                        keep_cols           = ["altura-1", "peso-1", "age", "hemoAd-4", "hemat-4",
                                               "inrAd-4", "gliceAd-4", "totalNIHSS-5", "preArtSis-5", "preArtDia-5", 
                                               NEW_COL, "time_since_onset", "altVis-5", "altCons-5", "ocEst-10"],
                        target_col          = "binary_rankin",
                        normalize           = True,
                        dirname             = DIR,
                        join_train_val      = True,
                        join_train_test     = True,
                        reshuffle           = True,
                        set_col             = "all",
                        filter_out_no_ncct  = False,
                        empty_values_method = missing)
    set  = loader.get_set("train")
    x, y = set["x"], np.array(set["y"])
    i = 0
    for train_index, test_index in StratifiedKFold(n_splits = 10, shuffle = False).split(x, y):
        i += 1
        print("fold", i)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for exp in EXPS:
            cols = EXPS[exp]
            name = f"ttest-{exp}"
            x_train_exp, x_test_exp = x_train[cols], x_test[cols]
            print("   ", missing, name, x_train_exp.shape, x_test_exp.shape)
            loader     = DummyLoader(x_train_exp, x_test_exp, y_train, y_test)
            classifier = logistic_regression(loader, n_iter = N_ITER, metric = METRIC, cv = CV)
            classifier.record_performance(f"{name}-{i}", missing, run_name = f"runs-{name}")
        name = "ttest-ASTRAL"
        x_train_exp, x_test_exp = x_train[ASTRAL_COLS].copy(), x_test[ASTRAL_COLS].copy()
        print("   ", missing, name, x_train_exp.shape, x_test_exp.shape)
        loader = DummyLoader(x_train_exp, x_test_exp, y_train, y_test)
        astral = ASTRALClassifier(dataset_filename = None, loader = loader)
        astral.record_performance(missing, run_name = f"runs-{name}")
        print()
        print()
