import os, numpy
import pandas as pd
from .dataset_splitter import SET, PATIENT_ID, BINARY_RANKIN
from .trainer import PERFORMANCE, PREDICTIONS
from .models import final_mlp
from abc import ABC, abstractmethod


class Ensemble(ABC):
    def __init__(self, experiments: list, experiments_dir: str = None, 
        data_dir: str = None, labels_filename: str = "dataset.csv"):
        self.experiments_dir = experiments_dir
        self.data_dir        = data_dir
        self.labels_filename = labels_filename
        self.load_experiments(experiments)
    def get_ytrue(self, preds_df: pd.DataFrame, set: str):
        if SET in preds_df.columns:
            predictions = preds_df[preds_df[SET] == set]
        else:
            assert set == "test"
            predictions = preds_df
        labels_df = self.labels_filename if self.data_dir is None else os.path.join(self.data_dir, self.labels_filename)
        labels_df = labels_df[labels_df[SET] == set]
        y         = []
        for _, row in predictions.iterrows():
            y.append( labels_df[labels_df[PATIENT_ID] == row[PATIENT_ID]][BINARY_RANKIN] )
        return y
    def get_best_run_preds(self, experiment):
        dir         = experiment if self.experiments_dir is None else os.path.join(experiments_dir, experiment)
        performance = pd.read_csv(os.path.join(dir, PERFORMANCE))
        best_run    = performance[performance[SET] == "val"]["f1_score"].argmax() + 1
        pred_df     = pd.read_csv(os.path.join(dir, f"{experiment}-run{best_run}", PREDICTIONS))
        return pred_df
    @abstractmethod
    def load_experiments(self, experiments):
        pass
    @abstractmethod        
    def get_probabilities(self):
        pass


class MajorityEnsemble(Ensemble):
    def load_experiments(self, experiments: list):
        self.predictions = {}
        for experiment in experiments:
            preds_df = self.get_best_run_preds(experiment)
            if SET in preds_df.columns:
                self.predictions[experiment] = preds_df[preds_df[SET] == "test"]["y_pred"].values
            else:
                self.predictions[experiment] = preds_df["y_pred"].values
    def get_probabilities(self):
        y_prob = None
        for experiment in self.predictions:
            if y_prob is None:
                y_prob = self.predictions[experiment]
            else:
                y_prob += self.predictions[experiment]
        return y_prob/len(self.predictions)

class AverageEnsemble(Ensemble):
    def load_experiments(self, experiments):
        self.probabilities = {}
        for experiment in experiments:
            preds_df = self.get_best_run_preds(experiment)
            if SET in preds_df.columns:
                self.probabilities[experiment] = preds_df[preds_df[SET] == "test"]["y_prob"].values
            else:
                self.probabilities[experiment] = preds_df["y_prob"].values
    def get_probabilities(self):
        y_prob = None
        for experiment in self.probabilities:
            if y_prob is None:
                y_prob = self.probabilities[experiment]
            else:
                y_prob += self.probabilities[experiment]
        return y_prob/len(self.probabilities)

class WeightedEnsemble(Ensemble):
    def load_experiments(self, experiments):
        self.sets = {"train": {"x": [], "y": []},
                     "val":   {"x": [], "y": []},
                     "test":  {"x": [], "y": []}}
        for set in self.sets:
            for experiment in experiments:
                preds_df = self.get_best_run_preds(experiment)
                assert SET in preds_df.columns, "WeightedEnsemble.load_experiments:"
                "only available for runs whose val and train sets probabilities are also stored."
                self.sets[set]["x"].append( preds_df[preds_df[SET] == set]["y_prob"].values )
                self.sets[set]["y"].append( self.get_ytrue(preds_df, set) )
            self.sets[set]["x"] = numpy.stack(self.sets[set]["x"], axis = 0)
            self.sets[set]["y"] = numpy.stack(self.sets[set]["y"], axis = 0)
    def fit(self, epochs = 100):
        self.lr = final_mlp( len(self.probabilities) )
    def get_probabilities(self):
        ...
