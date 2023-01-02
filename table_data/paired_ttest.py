import os, pandas as pd, scipy.stats as stats, numpy as np

'''
see
https://www.statology.org/paired-samples-t-test/
https://www.jmp.com/en_nl/statistics-knowledge-portal/t-test/paired-t-test.html
'''

ALPHA   = 0.05
METRIC  = "auc"
DIR     = "../../../runs/table data/runs-ttest/"

# CONSTANTS
FIRST_COL = "run\other run"

if __name__ == "__main__":
    pvalues         = {FIRST_COL: []}
    significance    = {FIRST_COL: []}
    for run in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, run, "performance.csv")):
            pvalues[run]        = []
            significance[run]   = []
    for run in os.listdir(DIR):
        path     = os.path.join(DIR, run, "performance.csv")
        if os.path.isfile(path):
            data     = pd.read_csv(path)
            data     = data[data.set == "test"]
            data     = data[METRIC].values
            pvalues[FIRST_COL].append(run)
            significance[FIRST_COL].append(run)
            for other_run in os.listdir(DIR):
                other_path  = os.path.join(DIR, other_run, "performance.csv")
                if os.path.isfile(other_path):
                    other_data  = pd.read_csv(other_path)
                    other_data  = other_data[other_data.set == "test"]
                    other_data  = other_data[METRIC].values
                    assert len(other_data) == 10, f"Run {run} has {len(data)} values"
                    pvalue      = stats.ttest_rel(data, other_data).pvalue
                    pvalues[other_run].append(pvalue)
                    if np.isnan(pvalue):
                        significance[other_run].append("")
                    else:
                        significance[other_run].append(["❌","✅"][int(pvalue < ALPHA)])
    pd.DataFrame(pvalues).to_csv(f"pvalues-{METRIC}.csv", index = False)
    pd.DataFrame(significance).to_csv(f"significance-{METRIC}.csv", index = False)
