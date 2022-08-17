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
    for missing in ("amputate", "impute"):
        loader  = TableLoader("table_data.csv",
                            keep_cols           = stage,
                            target_col          = "binary_rankin",
                            normalize           = True,
                            dirname             = "../../../data/gravo",
                            join_train_val      = True,
                            reshuffle           = False,
                            set_col             = "all",
                            filter_out_no_ncct  = False,
                            # empty_values_method = "impute")
                            empty_values_method = missing)
        # for classifier in (knns, decision_trees, random_forests):
        # TODO adicionar ao nome dos classifiers exportados o tipo de "empty_values_method" usado
        
        # Variaveis para remover:
        # "disf-5", "disf-15", "disf-19", "recaTIC-11"
        gradient_boosting(loader, stage, missing, n_iter = 50)
