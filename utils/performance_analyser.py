import os
import pandas as pd
from dataset_splitter import SET

PERFORMANCE = "performance.csv"

def list_dirs(path):
    return [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path,dir))]

class PerformanceAnalyser:
    def __init__(self, runs_dir: str = None, 
        metrics: list = ["accuracy", "precision", "recall", "f1_score", "auc_score"]):
        self.runs_dir = "" if runs_dir is None else runs_dir
        self.metrics  = metrics
        
    def analyse(self):
        out = {"type": [], "model": []}
        for metric in self.metrics:
            out[f"{metric}_avg"] = []
            out[f"{metric}_std"] = []
        for model_type in list_dirs(self.runs_dir):
            models_dir = os.path.join(self.runs_dir, model_type)
            for model in list_dirs(models_dir):
                model_performance = os.path.join(self.runs_dir, model_type, model, PERFORMANCE)
                performance_df    = pd.read_csv(model_performance, sep = ";")
                out["type"] .append(model_type)
                out["model"].append(model)
                for metric in self.metrics:
                    out[f"{metric}_avg"].append( performance_df[performance_df[SET] == "test"][metric].mean() )
                    out[f"{metric}_std"].append( performance_df[performance_df[SET] == "test"][metric].std() )
        out = pd.DataFrame(out)
        out = out.sort_values(by = "f1_score_avg", ascending = False)
        out.to_csv(os.path.join(self.runs_dir, PERFORMANCE), index = False)
        
        
if __name__ == "__main__":
    analyser = PerformanceAnalyser(runs_dir = "../../../runs/systematic/")
    analyser.analyse()
