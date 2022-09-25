import os
from table_loader import TableLoader
from utils.classic_classifiers import *
from sklearn.model_selection import train_test_split


DIR = "../../../data/gravo"
DATASET = "table_data.csv"
N_ITER = 40
CV     = 5
METRIC = "f1"


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


for missing in ("amputate", "impute", "impute_mean"):
    # , "impute_constant"):
    print(missing)
    # stage,missing_values,model,set,f1_score,accuracy,precision,recall,auc
    loader  = TableLoader(DATASET,
                        keep_cols           = ["altura-1", "peso-1", "age", "hemoAd-4", "hemat-4",
                                               "inrAd-4", "gliceAd-4", "totalNIHSS-5", "preArtSis-5", "preArtDia-5", 
                                               NEW_COL],
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
    x, y = set["x"], set["y"]
    for i in range(100):
        print(i)
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size = 0.2, 
                                                            random_state = i, 
                                                            stratify = y)
        loader = DummyLoader(x_train, x_test, y_train, y_test)
        classifier = logistic_regression(loader, n_iter = N_ITER, metric = METRIC, cv = CV)
        classifier.record_performance(f"important_features+occlusion_pred2-{i}", missing, run_name = "runs-important_features+occlusion_pred2")
