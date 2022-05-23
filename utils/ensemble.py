import os
import pandas as pd
from .trainer import PERFORMANCE, PREDICTIONS
from abc import ABC, abstractmethod

MAJORITY = "majority"
AVERAGE  = "average"
WEIGHTED = "weighted"
VOTING   = (MAJORITY, AVERAGE, WEIGHTED)


class Ensemble(ABC):
    def __init__(self, experiments: list, voting: str, experiments_dir: str = None):
        assert voting in VOTING, "Ensemble.__init__: valid voting schemes include {}, {} and {}".format(*VOTING)
        self.voting          = voting
        self.experiments_dir = experiments_dir
        self.predictions     = {}
        self.probabilities   = {}
    def load_experiments(self, experiments):
        assert len(experiments) > 1, "Ensemble.load_experiments: supply at least 2 experiments"
        for experiment in experiments:
            self.predictions[experiment]   = {}
            self.probabilities[experiment] = {}
            dir         = experiment if self.experiments_dir is None else os.path.join(experiments_dir, experiment)
            performance = pd.read_csv(os.path.join(dir, PERFORMANCE))
            best_run    = performance[performance["set"] == "val"]["f1_score"].argmax() + 1
            pred_df     = pd.read_csv(os.path.join(dir, f"{experiment}-run{best_run}", PREDICTIONS))
            if self.voting == WEIGHTED:
                assert "set" in predictions.columns, f"Ensemble.load_experiments: {WEIGHTED} voting"
                " scheme is only available for runs whose val and train sets probabilities are also stored."
                self.predictions[experiment]["train"]   = pred_df[pred_df["set"] == "train"]["y_pred"].values
                self.predictions[experiment]["val"]     = pred_df[pred_df["set"] == "val"]["y_pred"].values
                self.probabilities[experiment]["train"] = pred_df[pred_df["set"] == "train"]["y_prob"].values
                self.probabilities[experiment]["val"]   = pred_df[pred_df["set"] == "val"]["y_prob"].values
            else:
                if "set" in predictions.columns:
                    self.predictions[experiment]["test"]   = pred_df[pred_df["set"] == "test"]["y_pred"].values
                    self.probabilities[experiment]["test"] = pred_df[pred_df["set"] == "test"]["y_prob"].values
                else:
                    self.predictions[experiment]["test"]   = pred_df["y_pred"].values
                    self.probabilities[experiment]["test"] = pred_df["y_prob"].values
    @abstractmethod        
    def get_probabilities(self):
        pass


class MajorityEnsemble(Ensemble):
    def __init__(self, experiments: list, voting: str, experiments_dir: str = None):
        super().__init__(experiments, MAJORITY, experiments_dir = experiments_dir)
    def get_probabilities(self):
        y_prob = None
        for experiment in self.predictions:
            if y_prob is None:
                y_prob = self.predictions[experiment]["test"]
            else:
                y_prob += self.predictions[experiment]["test"]
        return y_prob/len(self.predictions)

class AverageEnsemble(Ensemble):
    def __init__(self, experiments: list, voting: str, experiments_dir: str = None):
        super().__init__(experiments, AVERAGE, experiments_dir = experiments_dir)
    def get_probabilities(self):
        y_prob = None
        for experiment in self.probabilities:
            if y_prob is None:
                y_prob = self.probabilities[experiment]["test"]
            else:
                y_prob += self.probabilities[experiment]["test"]
        return y_prob/len(self.probabilities)

class WeightedEnsemble(Ensemble):
    def __init__(self, experiments: list, voting: str, experiments_dir: str = None):
        super().__init__(experiments, WEIGHTED, experiments_dir = experiments_dir)
    def get_probabilities(self):
        ...
