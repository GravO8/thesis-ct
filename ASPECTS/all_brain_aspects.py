import sys
sys.path.append("..")
from utils.classic_classifiers import *
from radiomics_loader import RadiomicsLoader

loader = RadiomicsLoader("ncct_radiomic_features.csv", 
                        keep_cols           = "all",
                        target              = "aspects",
                        binary              = True,
                        normalize           = True,
                        dirname             = "../../../data/gravo",
                        join_train_val      = True,
                        join_train_test     = False,
                        reshuffle           = False)

# knns(loader)
# decision_trees(loader)
# random_forests(loader)
# logistic_regression(loader)
gradient_boosting(loader)
