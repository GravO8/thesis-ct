import sys
sys.path.append("..")
from utils.classic_classifiers import *
from utils.table_classifier import TableClassifier
# from cnn_features_loader import CNNFeaturesLoader
from hybrid_loader import HybridLoader
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

missing = "amputate"

for cnn in ("resnet18", "resnet34", "resnet50", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"):
    print(cnn)
    # loader  = CNNFeaturesLoader(f"{cnn}.csv",
    #                     keep_cols           = "all",
    #                     target_col          = "binary_rankin",
    #                     normalize           = True,
    #                     dirname             = "../../../data/MIP",
    #                     join_train_val      = True,
    #                     join_train_test     = False,
    #                     reshuffle           = True,
    #                     set_col             = "all",
    #                     empty_values_method = missing)
    loader  = HybridLoader(f"{cnn}-hybrid.csv",
                        keep_cols           = STAGE_BASELINE,
                        target_col          = "binary_rankin",
                        normalize           = True,
                        dirname             = "../../../data/MIP",
                        join_train_val      = True,
                        join_train_test     = False,
                        reshuffle           = True,
                        set_col             = "all",
                        empty_values_method = missing)
    print()
    for classifier in (knn, random_forest, logistic_regression):
        # , decision_tree, gradient_boosting):
        trained_classifier = classifier(loader, n_iter = 50, metric = "f1")
        trained_classifier.record_performance(cnn, missing)
