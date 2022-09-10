import sys
sys.path.append("..")
from utils.classic_classifiers import *
from utils.table_classifier import TableClassifier
from hybrid_loader import HybridLoader
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
DIR = "../../../data/MIP"

for missing in ("impute", "impute_mean", "impute_constant", "amputate"):
    for cnn in ("resnet18", "resnet34", "resnet50", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"):
        print(missing, cnn)
        loader_complementar = TableLoader(f"{cnn}-hybrid-complementar.csv",
                            keep_cols           = STAGE_BASELINE,
                            target_col          = "binary_rankin",
                            normalize           = False,
                            dirname             = DIR,
                            join_train_val      = True,
                            join_train_test     = True,
                            set_col             = "all",
                            filter_out_no_ncct  = False,
                            empty_values_method = None,
                            verbose             = False)
        loader  = HybridLoader(f"{cnn}-hybrid.csv",
                            keep_cols           = STAGE_BASELINE,
                            target_col          = "binary_rankin",
                            normalize           = True,
                            dirname             = DIR,
                            join_train_val      = True,
                            join_train_test     = False,
                            reshuffle           = True,
                            filter_out_no_ncct  = False,
                            set_col             = "all",
                            empty_values_method = missing,
                            loader_complementar = loader_complementar)
        print()
        for classifier in (knn, decision_tree, random_forest, logistic_regression, gradient_boosting):
            trained_classifier = classifier(loader, n_iter = 50, metric = "f1")
            trained_classifier.record_performance(cnn, missing)
