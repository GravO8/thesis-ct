import sys, torch, torchio, time, json, numpy, os, gc
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from skmultiflow.lazy import KNNClassifier
from torch.utils.data import DataLoader
from .assert_datasets import AssertDatasets
from .pytorchtools import EarlyStopping, Logger
from .losses import SupConLoss
from enum import Enum
from abc import ABC, abstractmethod


def print_gc():
    # from https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

class TrainerMode(Enum):
    SINGLE = 1
    KFOLD = 2
    def __str__(self):
        return self.name

class Trainer(ABC):
    def __init__(self, ct_loader, loss_fn = torch.nn.BCELoss(reduction = "mean"), 
    optimizer: torch.optim.Optimizer = torch.optim.Adam, trace_fn: str = "log", 
    batch_size: int = 32, num_workers: int = 0, epochs: int = 75, patience: int = 20, 
    optimizer_args: dict = {}):
        '''
        Input:  trace_fn, string either "print" or "log"
                TODO
        '''
        assert trace_fn in ("print", "log"), "Trainer.__init__: trace_fn must be either 'print' or 'log'"
        self.cuda           = torch.cuda.is_available()
        self.ct_loader      = ct_loader
        self.optimizer      = optimizer
        self.loss_fn        = loss_fn
        self.trace_fn       = trace_fn
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.epochs         = epochs
        self.patience       = patience
        self.optimizer_args = optimizer_args
        self.supcon         = isinstance(self.loss_fn, SupConLoss)
        self.mode           = None
        self.model          = None
        self.model_name     = None
        self.train_optimizer = None
        if self.supcon:
            self.knn = KNNClassifier(n_neighbors = 5, max_window_size = 2000)
            assert self.batch_size > 1, "Trainer.__init__: Supervised Contrastive loss requires a batch sizer > 0."
        
    def json_summary(self):
        '''
        TODO
        '''
        if self.train_optimizer is None:
            params          = {}
            optimizer_dict  = None
        else:
            params          = self.train_optimizer.state_dict()["param_groups"][0]
            optimizer_dict  = {"name": str(type(self.train_optimizer)),
                               "lr": params["lr"],
                               "amsgrad": params["amsgrad"],
                               "weight_decay": params["weight_decay"],
                               "betas": params["betas"],
                               "eps": params["eps"]}
        train_dict          = {"batch_size": self.batch_size,
                               "num_workers": self.num_workers,
                               "epochs": self.epochs,
                               "patience": self.patience}
        loader_dict         = self.ct_loader.to_dict()
        loader_dict["mode"] = str(self.mode)
        if self.mode == TrainerMode.SINGLE: loader_dict["train_size"]   = self.train_size
        else:
            loader_dict["k"]    = self.k
            loader_dict["fold"] = self.fold
        dict                = {"name": self.model_name,
                               "model": self.model.to_dict(), 
                               "ct_loader": loader_dict,
                               "optimizer": optimizer_dict,
                               "loss_fn": str(self.loss_fn),
                               "train": train_dict}
        with open(f"{self.model_name}/summary.json", "w") as f:
            json.dump(dict, f, indent = 4)
            
    def set_loaders(self, train, validation, test):
        '''
        TODO
        '''
        self.train_loader       = DataLoader(train, 
                                            batch_size  = self.batch_size, 
                                            num_workers = self.num_workers, 
                                            pin_memory  = self.cuda)
        self.validation_loader  = DataLoader(validation, 
                                            batch_size  = self.batch_size, 
                                            num_workers = self.num_workers,
                                            pin_memory  = self.cuda)
        self.test_loader        = DataLoader(test, 
                                            batch_size  = self.batch_size, 
                                            num_workers = self.num_workers,
                                            pin_memory  = self.cuda)
    
    def single(self, train_size: float = 0.8):
        '''
        TODO
        '''
        assert 0 < train_size < 1
        self.mode               = TrainerMode.SINGLE
        self.train_size         = train_size
        train, validation, test = self.ct_loader.subject_dataset(train_size = train_size)
        self.set_loaders(train, validation, test)

    def k_fold(self, k: int = 5):
        '''
        TODO
        '''
        assert k > 0
        self.mode           = TrainerMode.KFOLD
        self.k              = k
        self.fold_generator = self.ct_loader.k_fold(k = self.k)
        self.fold           = 0
        self.next_fold()
        
    def next_fold(self):
        '''
        TODO
        '''
        assert self.mode == TrainerMode.KFOLD
        train, validation, test = next(self.fold_generator)
        self.fold              += 1
        self.set_loaders(train, validation, test)
        
    def set_optimizer_args(self, args: dict):
        '''
        TODO
        '''
        self.optimizer_args = args
        
    def assert_supcon(self, model):
        '''
        TODO
        '''
        if self.supcon:
            assert model.return_features, "Trainer.train: model must return features for loss SupCon"
        # else:
        #     assert not model.return_features, "Trainer.train: model can only return features for loss SupCon"
            
    def assert_model_loaded(self):
        '''
        TODO
        '''
        assert self.model is not None, "Trainer.assert_model_loaded: call method 'set_model' first"
        assert self.model_name is not None, "Trainer.assert_model_loaded: call method 'set_model' first"
            
    def set_model(self, model: torch.nn.Module, model_name: str):
        '''
        TODO
        '''
        assert self.mode is not None, "Trainer.train: call method 'single' or 'k_fold' first"
        self.assert_supcon(model)
        self.model      = model
        self.model_name = model_name if self.mode == TrainerMode.SINGLE else f"{model_name}-fold{self.fold}of{self.k}"
        if self.cuda:
            self.model.cuda()
        if not os.path.isdir(self.model_name):
            os.mkdir(self.model_name)
        self.set_trace_fn()
        self.weights = f"{self.model_name}/weights.pt"
            
    def set_trace_fn(self):
        '''
        TODO
        '''
        if self.trace_fn == "log":
            logger      = Logger(f"{self.model_name}/log.txt")
            self.trace  = lambda s: logger.log(s)
        else:
            self.trace = print
        
    def train(self):
        '''
        TODO
        '''
        self.assert_model_loaded()
        self.train_optimizer    = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self.writer             = SummaryWriter(self.model_name)
        early_stopping          = EarlyStopping(patience    = self.patience, 
                                                verbose     = True, 
                                                path        = self.weights,
                                                trace_func  = self.trace,
                                                delta       = .000001)
        self.json_summary()
        self.time_start = time.time()
        self.trace(f"Using {'cuda' if self.cuda else 'CPU'} device")
        for epoch in range(self.epochs):
            self.trace(f"{self.model_name} - epoch {epoch}/{self.epochs} --------------------------------")
            val_loss    = self.train_validate_epoch(epoch)
            improved    = early_stopping(val_loss, self.model)
            if improved:
                self.best = self.current_scores
            if early_stopping.early_stop:
                self.trace("-==<[ Early stopping ]>==-")
                break
        with open(f"{self.model_name}/scores.json", "w") as f:
            json.dump(self.best, f, indent = 4) 
        self.model.load_state_dict(torch.load(self.weights)) # when done training, load best weights
        
    def train_validate_epoch(self, epoch: int):
        '''
        TODO
        '''
        train_loss, train_error = self.train_epoch()
        val_loss, val_error     = self.validate_epoch()
        elapsed                 = (time.time() - self.time_start)
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Error/train", train_error, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Error/val", val_error, epoch)
        self.writer.add_scalar("time", elapsed, epoch)
        self.writer.flush()
        self.trace(f"  Train loss: {train_loss}")
        self.trace(f"  Train error: {train_error}")
        self.trace(f"  Val loss: {val_loss}")
        self.trace(f"  Val error: {val_error}")
        self.trace(f"  time: {elapsed}")
        self.current_scores = {"train loss": train_loss,
                               "train error": train_error,
                               "val loss": val_loss,
                               "val error": val_error}
        return val_loss
        
    def train_epoch(self):
        '''
        Behaviour:  Applies all examples of the train set to the model and updates 
                    its weights according to the computed derivatives
        Output:     two real numbers with the train loss and error, respectivly
        '''
        self.model.train(True)
        train_loss  = 0
        train_error = 0
        if self.supcon:
            self.knn.reset()
        for subjects in self.train_loader:
            self.train_optimizer.zero_grad()  # reset gradients
            loss, error  = self.compute_loss_error(subjects, validate = False)
            loss.backward()             # compute the loss and its gradients
            self.train_optimizer.step()       # adjust learning weights
            train_loss  += float(loss)
            train_error += float(error)
        train_loss  /= len(self.train_loader)
        train_error /= len(self.train_loader)
        return train_loss, train_error
    
    def validate_epoch(self):
        '''
        Behaviour:  Applies all examples of the validation set to the model and
                    computes the loss and error
        Output:     two real numbers with the validation loss and error, respectivly
        '''
        self.model.train(False)
        val_loss    = 0
        val_error   = 0
        for subjects in self.validation_loader:
            loss, error = self.compute_loss_error(subjects, verbose = True, validate = True)
            val_loss   += float(loss)
            val_error  += float(error)
        val_loss       /= len(self.validation_loader)
        val_error      /= len(self.validation_loader)
        return val_loss, val_error
        
    def compute_loss_error(self, subjects, verbose: bool = False, validate: bool = False):
        '''
        TODO
        '''
        scans, y = self.get_batch(subjects)
        if self.supcon:
            features    = self.evaluate_brain(scans, verbose = verbose)
            loss        = self.loss_fn(features, y)
            features    = features.detach().cpu().numpy()
            if not validate:
                self.knn.partial_fit(features, y)
            y_prob  = self.knn.predict_proba(features)[:,1]
        else:
            y_prob  = self.evaluate_brain(scans, verbose = verbose)
            y_prob  = y_prob.squeeze().clamp(min = 1e-5, max = 1.-1e-5).cpu()
            loss    = self.loss_fn(y_prob, y)
            y_prob  = y_prob.detach().numpy() 
        y_pred = (y_prob > .5).astype(int)
        error = (1. - numpy.equal(y_pred, y.cpu().numpy()).astype(int)).sum()/len(y)
        if verbose:
            for i in range(len(y)):
                self.trace(f" - True label: {int(y[i])}. Predicted: {int(y_pred[i])}. Probability: {round(float(y_prob[i]), 4)}")
        return loss, error
    
    def get_batch(self, subjects):
        '''
        TODO
        '''
        scans   = subjects["ct"][torchio.DATA]
        y       = subjects["target"].float()
        if self.cuda:
            scans = scans.cuda()
        return scans, y
        
    def init_knn(self):
        '''
        TODO
        assumes model is loaded
        Resets the KNN and loads the examples from the train and val sets into 
        this classifier 
        '''
        self.knn = KNNClassifier(n_neighbors = 5, max_window_size = 2000)
        for subjects in self.train_loader:
            scans, y = self.get_batch(subjects)
            features = self.evaluate_brain(scans, verbose = False)
            features = features.detach().cpu().numpy()
            self.knn.partial_fit(features, y)
        for subjects in self.validation_loader:
            scans, y = self.get_batch(subjects)
            features = self.evaluate_brain(scans, verbose = False)
            features = features.detach().cpu().numpy()
            self.knn.partial_fit(features, y)
        
    def test(self, t, verbose = True):
        '''
        TODO
        '''
        self.assert_model_loaded()
        self.model.train(False)
        if self.supcon:
            self.init_knn()
        ys, y_preds = [], []
        for subjects in self.test_loader:
            scans, y = self.get_batch(subjects)
            if self.supcon:
                features    = self.evaluate_brain(scans, verbose = verbose)
                features    = features.detach().cpu().numpy()
                y_pred      = self.knn.predict( features )
            else:
                y_prob  = self.evaluate_brain(scans, verbose = verbose)
                y_prob  = y_prob.squeeze().clamp(min = 1e-5, max = 1.-1e-5).cpu()
                y_prob  = y_prob.detach().numpy() 
                y_pred  = (y_prob > .5).astype(int)
            ys.extend( [int(r) for r in list(y)] )
            y_preds.extend( [int(r) for r in y_pred.tolist()] )
        auc         = metrics.roc_auc_score  (y_true = ys, y_score = y_preds)
        accur       = metrics.accuracy_score (y_true = ys, y_pred = y_preds)
        recall      = metrics.recall_score   (y_true = ys, y_pred = y_preds)
        precision   = metrics.precision_score(y_true = ys, y_pred = y_preds)
        if verbose:
            self.trace(" Accuracy:\t{:.2f}".format(accur))
            self.trace(" AUC:\t\t{:.2f}".format(auc))
            self.trace(" Recall:\t{:.2f}".format(recall))
            self.trace(" Precision:\t{:.2f}".format(precision))
        scores = {"accuracy": accur, "AUC": auc, "recall": recall, "precision": precision}
        with open(f"{self.model_name}/scores-test-{t}.json", "w") as f:
            json.dump(scores, f, indent = 4)
            
    def save_encodings(self):
        '''
        Saves the encodings of a given set to disk
        TODO
        '''
        def save_encodings_aux(subjects_set, subjects_loader):
            s           = 0
            labels      = []
            dir_name    = f"{encodings_dir}/{subjects_set}"
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            for subjects in subjects_loader:
                scans, y = self.get_batch(subjects)
                features = self.evaluate_brain(scans, verbose = False)
                for i in range(len(y)):
                    labels.append(int(y[i]))
                    torch.save(features[i,:].squeeze(), f"{encodings_dir}/{subjects_set}/subject-{s}.pt")
                    s += 1
            torch.save(torch.tensor(labels), f"{encodings_dir}/{subjects_set}/labels.pt")
        self.assert_model_loaded()
        encodings_dir = f"{self.model_name}/encodings"
        if not os.path.isdir(encodings_dir):
            os.mkdir(encodings_dir)
        self.model.train(False)
        save_encodings_aux("train", self.train_loader)
        save_encodings_aux("validation", self.validation_loader)
        save_encodings_aux("test", self.test_loader)
        
    def assert_datasets(self):
        '''
        TODO
        '''
        asserter = AssertDatasets(  self.train_loader, 
                                    self.validation_loader, 
                                    self.test_loader)
        asserter.assert_leaks()
        asserter.assert_repeated()
        if self.ct_loader.balance_train_set:
            asserter.assert_balanced("train")
            asserter.assert_balanced("validation")
        if self.ct_loader.balance_test_set:
            asserter.assert_balanced("test")

    @abstractmethod
    def evaluate_brain(self, subjects, verbose: bool = False):
        pass


class MILTrainer(Trainer):
    def evaluate_brain(self, scans, verbose: bool = False):
        '''
        TODO
        '''
        out = []
        for scan in scans.unbind(dim = 0):
            shp     = scan.shape
            # (Z, B, W, H) where B is actually the number of channels now and Z the batch size
            scan    = scan.permute((3,0,1,2))
            # filter slices that have mostly 0s
            scan    = scan[[i for i in range(scan.shape[0]) if torch.count_nonzero(scan[i,:,:,:] > 0) > 100]]
            out.append( self.model(scan) )
        return torch.stack(out, dim = 0) 


class SiameseTrainer(Trainer):
    def evaluate_brain(self, scans, verbose: bool = False):
        '''
        TODO
        '''
        msp         = scans.shape[2]//2             # midsagittal plane
        hemisphere1 = scans[:,:,:msp,:,:]           # shape = (B,C,x,y,z)
        hemisphere2 = scans[:,:,msp:,:,:].flip(2)   # B - batch; C - channels
        return self.model(hemisphere1, hemisphere2)
