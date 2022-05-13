import torch, torchio
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

LR          = 0.0005 # learning rate
WD          = 0.0001 # weight decay
LOSS        = torch.nn.BCELoss(reduction = "mean")
OPTIMIZER   = torch.optim.Adam


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
        train, validation, test = ct_loader.load_dataset()
        self.train_loader       = torch.utils.data.DataLoader(train, 
                                            batch_size  = self.batch_size, 
                                            num_workers = self.num_workers, 
                                            pin_memory  = self.cuda)
        self.validation_loader  = torch.utils.data.DataLoader(validation, 
                                            batch_size  = self.batch_size, 
                                            num_workers = self.num_workers,
                                            pin_memory  = self.cuda)
        self.test_loader        = torch.utils.data.DataLoader(test, 
                                            batch_size  = self.batch_size, 
                                            num_workers = self.num_workers,
                                            pin_memory  = self.cuda)

    def train(self, model):
        self.train_optimizer = OPTIMIZER(model.parameters(), 
                                                lr = LR,
                                                weight_decay = WD)
        # self.trace(f"Using {'cuda' if self.cuda else 'CPU'} device")
        self.best_score = None
        for epoch in range(self.epochs):
            train_metrics   = self.train_epoch()
            val_metrics     = get_metrics( self.get_probabilities(self.validation_loader) )
            test_metrics    = get_metrics(self.tset_loader)
            self.save_metrics(epoch, train_metrics, val_metrics, test_metrics, verbose = True)
            self.save_weights(val_metrics["f1_score"])
            
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
            self.trace_fn(row.format("", "loss", "f1-score", "accuracy"))
            for set in metrics:
                self.trace_fn(row.format(set, round(metrics[set]["loss"],2), 
                round(metrics[set]["f1-score"],2), round(metrics[set]["accuracy"],2)))
                
    def save_weights(self, val_f1_score, verbose = True):
        if (self.best_score is None) or (val_f1_score > self.best_score):
            if verbose:
                self.trace_func(f"Validation f1-score increased ({self.best_score} --> {val_f1_score}).  Saving model ...")
            self.best_score = val_f1_score
            torch.save(self.model.state_dict(), self.path)
    
    
if __name__ == '__main__':
    from ct_loader import CTLoader
    ct_loader = CTLoader(data_dir = "../../../data/gravo")
    trainer = Trainer(ct_loader)
