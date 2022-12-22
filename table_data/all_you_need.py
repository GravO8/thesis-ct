import os
from table_loader import TableLoader
from utils.classic_classifiers import *
from sklearn.model_selection import train_test_split


DIR         = "../../../data/gravo"
DATASET     = "table_data.csv"
N_ITER      = 40
CV          = 5
METRIC      = "f1"
RUN_NAME    = "important_features+occlusion+aspects+leukoaraiosis"


# NEW_COL         = "occlusion-pred2"
table_data      = pd.read_csv(os.path.join(DIR,DATASET))
# occlusion_pred  = pd.read_csv("dataset-occlusion.csv")
# preds           = []
# paint           = []
# experiment where the patients whose CTAs are not available (and hence, whose 
# predictions are also not available), have their occlusion variable given by 
# the GT
# paint can have three values:
#   1   when a prediction is available
#   0   when a prediction is not available but the GT is
#   -1  when neither the prediction nor the GT are available
# (after calling `get_set` no row has paint = -1 as at least one occlusion value
# for each patient should be available)
# for _, row in table_data.iterrows():
#     pred = occlusion_pred[occlusion_pred.patient_id.astype(str) == row.idProcessoLocal]["occlusion-pred"].values
#     if len(pred) > 0:
#         preds.append(int(pred[0] > .5))
#         paint.append(1)
#     else:
#         gt = row["ocEst-10"]
#         if gt != "None":
#             preds.append(gt)
#             paint.append(0)
#         else:
#             preds.append("")
#             paint.append(-1)
# table_data[NEW_COL] = preds
# table_data["paint"] = paint
# table_data.to_csv(os.path.join(DIR,DATASET), index = False)


class DummyLoader:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.sets = {"train": {"x": x_train, "y": y_train}, 
                    "test":   {"x": x_test, "y": y_test}}
    def available_sets(self):
        return [s for s in self.sets]
    def get_set(self, set):
        return self.sets[set]


for missing in ("amputate", ):
    # "impute", "impute_mean"):
    # , "impute_constant"):
    print(missing)
    # stage,missing_values,model,set,f1_score,accuracy,precision,recall,auc
    loader  = TableLoader(DATASET,
                        keep_cols           = [ "age", "totalNIHSS-5", "time_since_onset", 
                                                "altVis-5", "altCons-5", "gliceAd-4", 
                                                "ocEst-10", "aspects-7", "leucoa-7", 
                                                "occlusion-pred2"],
                        target_col          = "binary_rankin",
                        normalize           = False,
                        dirname             = DIR,
                        join_train_val      = True,
                        join_train_test     = True,
                        reshuffle           = True,
                        set_col             = "all",
                        filter_out_no_ncct  = False,
                        empty_values_method = missing)
    set  = loader.get_set("train")
    x, y = set["x"], set["y"]
    exit()
    set["x"]["leucoa-7"] = [1 if v>0 else 0 for v in set["x"]["leucoa-7"].values]
    # paint = x["paint"]
    # import numpy as np
    # print(np.unique(paint, return_counts = 1))
    # print(np.unique(y, return_counts = 1))
    for i in range(100):
        print(i)
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size = 0.2, 
                                                            random_state = i, 
                                                            stratify = y)
        loader = DummyLoader(x_train, x_test, y_train, y_test)
        classifier = logistic_regression(loader, n_iter = N_ITER, metric = METRIC, cv = CV)
        classifier.record_performance(f"{RUN_NAME}-{i}", missing, run_name = f"runs-{RUN_NAME}")
