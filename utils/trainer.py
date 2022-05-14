import torch, torchio
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from logger import Logger

LR          = 0.0005 # learning rate
WD          = 0.0001 # weight decay
LOSS        = torch.nn.BCELoss(reduction = "mean")
OPTIMIZER   = torch.optim.Adam

PERFORMANCE = "performance.csv"
PREDICTIONS = "predictions.csv"


def compute_metrics(y_true, y_prob):
    y_pred      = (y_prob > .5).astype(int)
    loss        = LOSS(y_prob, y)
    accuracy    = metrics.accuracy_score    (y_true = y_true, y_pred = y_pred)
    precision   = metrics.precision_score   (y_true = y_true, y_pred = y_pred)
    recall      = metrics.recall_score      (y_true = y_true, y_pred = y_pred)
    f1_score    = metrics.f1_score          (y_true = y_true, y_pred = y_pred)
    auc         = metrics.roc_auc_score     (y_true = y_true, y_score = y_prob)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, 
            "f1-score": f1_score, "auc": auc}


class Trainer:
    def __init__(self, ct_loader, batch_size: int = 32, epochs: int = 300):
        if torch.cuda.is_available():
            self.cuda           = True
            self.num_workers    = 8
            self.trace_fn       = "log"
        else:
            self.cuda           = False
            self.num_workers    = 0
            self.trace_fn       = "print"
        self.batch_size = batch_size
        self.set_loaders(ct_loader)

    def set_loaders(self, ct_loader):
        '''
        TODO
        '''
        train, val, test = ct_loader.load_dataset()
        self.train_loader = torch.utils.data.DataLoader(train, 
                                    batch_size  = self.batch_size, 
                                    num_workers = self.num_workers, 
                                    pin_memory  = self.cuda)
        self.val_loader   = torch.utils.data.DataLoader(val, 
                                    batch_size  = self.batch_size, 
                                    num_workers = self.num_workers,
                                    pin_memory  = self.cuda)
        self.test_loader  = torch.utils.data.DataLoader(test, 
                                    batch_size  = self.batch_size, 
                                    num_workers = self.num_workers,
                                    pin_memory  = self.cuda)
                                            
    def set_train_model(self, model):
        self.model = model
        model_name = self.model.get_name()
        if not os.path.isdir(model_name):
            os.system(f"mkdir {model_name}")
            with open(os.path.join(model_name, PERFORMANCE), "w") as f:
                f.write("model_name;run;set;best_epoch;accuracy;precision;recall;f1_score;auc_score\n")
            with open(os.path.join(model_name, "summary.txt"), "w") as f:
                f.write( str(self.model) )
        prev_runs   = [f for f in os.listdir(model_name) if f.startswith(model_name)]
        self.run    = 1 + len(prev_runs)
        run_dir     = os.path.join(model_name, f"{model_name}-run{self.run}")
        os.system(f"mkdir {run_dir}")
        if self.trace_fn == "log":
            logger      = Logger(os.path.join(run_dir, "log.txt"))
            self.trace  = lambda s: logger.log(s)
        else:
            self.trace = print
        self.weights_path = os.path.join(run_dir, "weights.pt")
        self.writer       = SummaryWriter(run_dir)
        
    def reset_model(self):
        self.model        = None
        self.best_score   = None
        self.best_epoch   = None
        self.trace        = None
        self.weights_path = None
        self.writer       = None
        self.run          = None

    def train(self, model):
        self.set_train_model(model)
        self.train_optimizer = OPTIMIZER(self.model.parameters(), 
                                        lr = LR,
                                        weight_decay = WD)
        self.trace(f"Using {'cuda' if self.cuda else 'CPU'} device")
        for epoch in range(self.epochs):
            train_metrics   = self.train_epoch()
            val_metrics     = get_metrics( self.get_probabilities(self.val_loader) )
            test_metrics    = get_metrics( self.get_probabilities(self.test_loader) )
            self.save_metrics(epoch, train_metrics, val_metrics, test_metrics, verbose = True)
            self.save_weights(val_metrics["f1_score"], epoch)
        self.record_performance()
        self.reset_model()
            
    def train_epoch(self):
        self.trace(f"{self.model_name} - epoch {epoch}/{self.epochs} --------------------------------")
        self.model.train(True)
        y_trues, y_probs = [], []
        for batch in self.train_loader:
            self.train_optimizer.zero_grad()  # reset gradients
            x, y_true   = self.get_batch(batch)
            y_prob      = self.model(x)
            y_prob      = y_prob.squeeze().clamp(min = 1e-5, max = 1.-1e-5).cpu()
            loss        = LOSS(y_prob, y_true)
            loss.backward()                   # compute the loss and its gradients
            self.train_optimizer.step()       # adjust learning weights
            y_trues.append(y_true)
            y_probs.append(y_prob.detach().numpy())
        y_true  = np.concatenate(y_trues, axis = 0)
        y_prob  = np.concatenate(y_probs, axis = 0)
        return compute_metrics(y, y_prob)
        
    def get_probabilities(self, set_loader):
        self.model.train(False)
        y_trues, y_probs = [], []
        for batch in set_loader:
            x, y_true   = self.get_batch(batch)
            y_prob      = self.model(x)
            y_prob      = y_prob.squeeze().clamp(min = 1e-5, max = 1.-1e-5).cpu()
            y_prob      = y_prob.detach().numpy()
            y_trues.append(y_true)
            y_probs.append(y_prob.detach().numpy())
        y_true  = np.concatenate(y_trues, axis = 0)
        y_prob = np.concatenate(y_probs, axis = 0)
        return y_true, y_prob
            
    def get_batch(self, subjects):
        scans   = subjects["ct"][torchio.DATA]
        y       = subjects["binary_rankin"].float()
        if self.cuda:
            scans = scans.cuda()
        return scans, y
        
    def tensorboard_metrics(self, epoch: int, metrics: dict):
        set_names    = ("train", "val", "test")
        metric_names = ("loss", "f1-score", "accuracy")
        for set in metrics:
            for metric in metric_names:
                self.writer.add_scalar(f"{metric}/{set}", metrics[set][metric], epoch)
        self.writer.flush()
        
    def save_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict, 
        test_metrics: dict, verbose: bool = True):
        metrics = {"train": train_metrics, "val": val_metrics, "test": test_metrics}
        self.tensorboard_metrics(epoch, metrics)
        if self.verbose:
            row = "{:<10}"*4
            self.trace(row.format("", "loss", "f1-score", "accuracy"))
            for set in metrics:
                self.trace(row.format(set, round(metrics[set]["loss"],2), 
                round(metrics[set]["f1-score"],2), round(metrics[set]["accuracy"],2)))
                
    def save_weights(self, val_f1_score, epoch, verbose = True):
        if (self.best_score is None) or (val_f1_score > self.best_score):
            if verbose:
                self.trace(f"Validation f1-score increased ({self.best_score} --> {val_f1_score}).  Saving model ...")
            self.best_score = val_f1_score
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.weights_path)
            
    def record_performance(self):
        self.model.load_state_dict(torch.load(self.weights_path))
        model_name       = self.model.get_name()
        metrics          = {}
        metrics["train"] = get_metrics( self.get_probabilities(self.train_loader) )
        metrics["val"]   = get_metrics( self.get_probabilities(self.val_loader) )
        y_test, y_prob   = self.get_probabilities(self.test_loader)
        y_pred           = (y_prob > .5).astype(int)
        metrics["test"]  = get_metrics(y_test, y_prob)
        performance_row  = f"{model_name};{self.run};{self.best_epoch}" + (";{}"*6) + "\n"
        test_ids         = [int(patient_id) for batch in self.test_loader for patient_id in batch]
        with open(os.path.join(model_name, PERFORMANCE), "a") as f:
            for set in metrics:
                f.write(performance_row.format(set, metrics[set]["accuracy"], 
                metrics[set]["precision"], metrics[set]["recall"], 
                metrics[set]["f1_score"], metrics[set]["auc"]))
        with open(os.path.join(model_name, f"{model_name}-run{self.run}", PREDICTIONS), "w") as f:
            f.write("patiend_id;y_prob;y_pred\n")
            for i in range(len(test_ids)):
                f.write(f"{test_ids[i]};{y_prob[i]};{y_pred[i]}\n")
    
    
if __name__ == '__main__':
    from ct_loader import CTLoader
    ct_loader = CTLoader(data_dir = "../../../data/gravo")
    trainer = Trainer(ct_loader)
    # a = [int(patient_id) for batch in trainer.test_loader for patient_id in batch["patient_id"]]
    # b = [int(patient_id) for batch in trainer.test_loader for patient_id in batch["patient_id"]]
    # print(a == b)
