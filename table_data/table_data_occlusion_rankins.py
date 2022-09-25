import sys, os
sys.path.append("..")
from utils.classic_classifiers import *
from utils.table_classifier import TableClassifier
from table_loader import TableLoader
from stages import *

DIR = "../../../data/gravo"
DATASET = "table_data.csv"

NEW_COL         = "occlusion-pred"
table_data      = pd.read_csv(os.path.join(DIR,DATASET))
occlusion_pred  = pd.read_csv("dataset-occlusion.csv")
preds           = []
for _, row in table_data.iterrows():
    pred = occlusion_pred[occlusion_pred.patient_id.astype(str) == row.idProcessoLocal][NEW_COL].values
    if len(pred) > 0:
        preds.append(pred[0])
    else:
        preds.append("")
table_data[NEW_COL] = preds
table_data.to_csv(os.path.join(DIR,DATASET), index = False)


for missing in ("impute",):
     # "amputate", "impute", "impute_mean", "impute_constant"):
    print(missing)
    # stage,missing_values,model,set,f1_score,accuracy,precision,recall,auc
    loader  = TableLoader(DATASET,
                        keep_cols           = ["altura-1", "peso-1", "age", "hemoAd-4", "hemat-4",
                                               "inrAd-4", "gliceAd-4", "totalNIHSS-5", "preArtSis-5", "preArtDia-5", 
                                               NEW_COL],
                                                # ocEst-10 ], totalNIHSS-5
                        target_col          = "binary_rankin",
                        normalize           = True,
                        dirname             = DIR,
                        join_train_val      = True,
                        join_train_test     = False,
                        reshuffle           = True,
                        set_col             = "all",
                        filter_out_no_ncct  = False,
                        empty_values_method = missing)
    print()
    # trained_classifier = logistic_regression(loader, 
    #                                         n_iter = 50, 
    #                                         metric = "f1", 
    #                                         # weights = "runs/models/LogisticRegression-ocEst-10-impute.joblib")
    #                                         weights = f"../../../runs/table data/runs-important_features-orig/runs-important_features+occlusion_pred/models/LogisticRegression-occlusion-pred-impute.joblib")
    # trained_classifier.record_pretrained_performance(NEW_COL, missing)
    trained_classifier = logistic_regression(loader, n_iter = 50, metric = "f1")
    trained_classifier.record_performance(NEW_COL, missing)
