import sys
sys.path.append("..")
from utils.classic_classifiers import *
from utils.table_classifier import TableClassifier
from table_loader import TableLoader
from stages import *

# stage   = STAGE_BASELINE
# missing = "amputate"
# loader  = TableLoader("table_data.csv",
#                     keep_cols           = stage,
#                     target_col          = "binary_rankin",
#                     normalize           = True,
#                     dirname             = "../../../data/gravo",
#                     join_train_val      = True,
#                     reshuffle           = False,
#                     set_col             = "all",
#                     filter_out_no_ncct  = False,
#                     # empty_values_method = "impute")
#                     empty_values_method = missing)

for stage in STAGES:
    for missing in ("amputate", "impute", "impute_mean", "impute_constant"):
        print(stage, missing)
        # stage,missing_values,model,set,f1_score,accuracy,precision,recall,auc
        loader  = TableLoader("table_data.csv",
                            keep_cols           = ["altura-1", "peso-1", "age", "hemoAd-4", "hemat-4",
                                                   "inrAd-4", "gliceAd-4", "totalNIHSS-5", "preArtSis-5", 
                                                   "preArtDia-5"],
                            target_col          = "binary_rankin",
                            normalize           = True,
                            dirname             = "../../../data/gravo",
                            join_train_val      = True,
                            join_train_test     = False,
                            reshuffle           = True,
                            set_col             = "all",
                            filter_out_no_ncct  = False,
                            empty_values_method = missing)
        print()
        loader.save_set_splits()
        exit(0)
        for classifier in [logistic_regression]:
         # (knn, decision_tree, random_forest, logistic_regression, gradient_boosting):
            trained_classifier = classifier(loader, n_iter = 50, metric = "f1")
            trained_classifier.record_performance(stage, missing)
