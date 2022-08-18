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
        loader  = TableLoader("table_data.csv",
                            keep_cols           = stage,
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
        for classifier in (knn, decision_tree, random_forest, logistic_regression, gradient_boosting, svm):
            trained_classifier = classifier(loader, n_iter = 50)
            trained_classifier.record_performance(stage, missing)
