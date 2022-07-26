import sys
sys.path.append("..")
from utils.table_classifier import *
from table_loader import TableLoader
from stages import STAGE_BASELINE


loader = TableLoader("table_data.csv", 
                    keep_cols           = STAGE_BASELINE,
                    target_col          = "binary_rankin",
                    normalize           = True,
                    dirname             = "../../../data/gravo",
                    join_train_val      = True,
                    reshuffle           = False,
                    # empty_values_method = "impute")
                    empty_values_method = "amputate")
# knns(loader)
# decision_trees(loader)
# random_forests(loader)
logistic_regression(loader)
# gradient_boosting(loader)
