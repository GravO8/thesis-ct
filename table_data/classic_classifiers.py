import sys
sys.path.append("..")
from utils.table_classifier import knns
from table_loader import TableLoader
from stages import STAGE_BASELINE


loader = TableLoader("table_data.csv", 
                    keep_cols   = STAGE_BASELINE,
                    target_col  = "binary_rankin",
                    normalize   = True,
                    dirname     = "../../../data/gravo",
                    join_train_val = True)
knns(loader)
