from table_loader import TableLoader
from stages import *
import pandas as pd, os, numpy as np


DIR = "../../../tese/table_data"


def percentage(num, total):
    return num*100/total


def five_num_summary(data):
    min, max = data.min(), data.max()
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    return np.array([min, q1, median, q3, max])


def analyse_stage(cols: list, stage_info: pd.DataFrame):
    missing = []
    distr   = []
    total   = len(loader.sets["train"]["x"])
    for _, row in stage_info.iterrows():
        var_name = row["variable"]
        assert var_name in cols
        type = row["type"]
        data = loader.sets["train"]["x"][var_name].values
        if type == "boolean":
            labels, counts    = np.unique(data, return_counts = True)
            zeros, ones, miss = counts
            distr.append( f"0:{zeros} ({percentage(zeros,total):.1f}%)  1:{ones} ({percentage(ones,total):.1f}%)" )
            missing.append( f"{miss} ({percentage(miss,total):.1f}%)" )
        elif type == "categorical":
            labels, counts = np.unique(data, return_counts = True)
            counts, miss   = counts[:-1], counts[-1]
            labels         = [int(l) for l in labels if not np.isnan(l) and l==int(l)]
            d = ""
            for i in range(len(counts)):
                d += f"{labels[i]}:{counts[i]}  "
            distr.append(d[:-2])
            missing.append( f"{miss} ({percentage(miss,total):.1f}%)" )
        elif (type == "integer") or (type == "real"):
            nans = np.isnan(data)
            miss = np.count_nonzero(nans)
            data = data[~nans]
            summ = five_num_summary(data)
            if type == "integer":
                summ = summ.astype(int)
                distr.append( f"{summ[0]}  {summ[1]}  {summ[2]}  {summ[3]}  {summ[4]}" )
            else:
                distr.append( f"{summ[0]:.2f}  {summ[1]:.2f}  {summ[2]:.2f}  {summ[3]:.2f}  {summ[4]:.2f}" )
            missing.append( f"{miss} ({percentage(miss,total):.1f}%)" )
        else:
            missing.append(np.nan)
            distr.append(np.nan)
    stage_info["missing_entries"] = missing
    stage_info["distribution"] = distr
    return stage_info


if __name__ == "__main__":
    loader  = TableLoader("table_data.csv",
                        keep_cols           = STAGE_PRETREATMENT,
                        target_col          = "binary_rankin",
                        normalize           = False,
                        dirname             = "../../../data/gravo",
                        join_train_val      = True,
                        join_train_test     = True,
                        reshuffle           = False,
                        set_col             = "all",
                        filter_out_no_ncct  = False,
                        empty_values_method = None)
    # a = loader.sets["train"]["x"]["hemat-4"].unique()
    # a.sort()
    # print( a )
    # exit(0)
                         
    # for stage in STAGES:
    #     stage_path = os.path.join(DIR, f"{stage}.csv")
    #     stage_info = pd.read_csv(stage_path)
    #     stage_info = analyse_stage(STAGES[stage], stage_info)
    #     stage_path = stage_path.split(".csv")[0] + "-stats.csv"
    #     stage_info.to_csv(stage_path, index = False)
