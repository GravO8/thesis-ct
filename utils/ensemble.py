from .dataset_splitter import SET, PATIENT_ID, BINARY_RANKIN
from .trainer import LR, WD, LOSS, OPTIMIZER, PERFORMANCE, PREDICTIONS, compute_metrics
from .models import final_mlp
from abc import ABC, abstractmethod

ENSEMBLE = "ensemble.csv"


def majority_ensemble(experiments: list, **kwargs):
    return Ensemble(experiments, probability = False, sets = ["test"], **kwargs)

def average_ensemble(experiments: list, **kwargs):
    return Ensemble(experiments, probability = True, sets = ["test"], **kwargs)
    
def weighted_average_ensemble(experiments: list, **kwargs):
    return Ensemble(experiments, probability = True, sets = ["train", "val", "test"], **kwargs)


class Ensemble:
    def __init__(self, experiments: list, probability: bool, sets: list, 
        experiments_dir: str = None, data_dir: str = None, 
        labels_filename: str = "dataset.csv"):
        assert len(experiments) > 1
        assert 1 <= len(sets) <= 3
        self.experiments     = experiments
        self.probability     = probability
        self.col             = "y_prob" if probability else "y_pred"
        self.experiments_dir = experiments_dir
        self.data_dir        = data_dir
        self.labels_filename = labels_filename
        self.mlp             = final_mlp(len(experiments), bias = False)
        self.load_experiments(sets)
        self.trainable       = len(sets) == 3
        if not self.trainable:
            with torch.no_grad():
                self.mlp[0].weight.fill_(1/len(experiments))
                
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
    
    def load_experiments(sets: list):
        self.sets = {set: {"x": [], "y": None} for set in sets}
        if ("train" in sets) or ("val" in sets):
            assert SET in preds_df.columns, "WeightedEnsemble.load_experiments:"
            "only available for runs whose val and train sets probabilities are also stored."
        for set in self.sets:
            for experiment in self.experiments:
                preds_df = self.get_best_run_preds(experiment)
                self.sets[set]["x"].append( preds_df[preds_df[SET] == set][self.col].values )
            self.sets[set]["x"] = torch.Tensor(numpy.stack(self.sets[set]["x"], axis = 0))
            self.sets[set]["y"] = torch.Tensor(self.get_ytrue(preds_df, set))
            
    def get_probabilities(self, set: str = "test"):
        return self.mlp(self.sets[set]["x"])
        
    def evaluate(self, set: str = "test"):
        y_pred = self.get_probabilities(set)
        y_true = self.sets[set]["y"]
        return compute_metrics(y_true, y_pred)
        
    def fit(self, epochs = 100):
        assert self.trainable
        train_optimizer = OPTIMIZER(self.mlp.parameters(), 
                                    lr = LR,
                                    weight_decay = WD)
        best_score = None
        for epoch in range(epochs):
            train_optimizer.zero_grad()  # reset gradients
            y_true  = self.sets["train"]["y"]
            y_prob  = self.get_probabilities("train")
            y_prob  = y_prob.squeeze().clamp(min = 1e-5, max = 1.-1e-5)
            loss    = LOSS(y_prob, y_true)
            loss.backward()              # compute the loss and its gradients
            train_optimizer.step()       # adjust learning weights
            metrics = self.evaluate("val")
            if (best_score is None) or (metrics["f1_score"] > best_score):
                best_score   = metrics["f1_score"]
                best_weights = self.mlp[0].weights.detach().numpy()
        with torch.no_grad():
            self.mlp[0].weight = torch.nn.Parameter(torch.from_numpy(best_weights).float())
            
    def record_performance(self, train: bool = True, epochs: int = 100):
        file = os.path.join(experiments_dir, ENSEMBLE)
        if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write("experiment_names;probability;weights;accuracy;precision;recall;f1_score;auc_score\n")
        if self.trainable and train:
            self.fit(epochs)
        metrics = self.evaluate("test")
        with open(file, "w") as f:
            f.write(f'"{self.experiments[0]}')
            for experiment in self.experiments[1:]:
                f.write(f",{experiment}")
            f.write(f'";{self.probability};"')
            for w in self.mlp[0].weight.detach().numpy()[0]:
                f.write(f",{w}")
            f.write('";{};{};{};{};{}\n'.format(metrics["accuracy"], metrics["precision"],
                metrics["recall"], metrics["f1-score"], metrics["auc"]))
