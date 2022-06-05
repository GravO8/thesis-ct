import sys, torch, torchio
import numpy as np
sys.path.append("..")
from utils.trainer import Trainer, compute_metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from lightly.loss.ntx_ent_loss import NTXentLoss


LOSS = NTXentLoss(memory_bank_size = 0)


class UnSupTrainer(Trainer):
    def __init__(self, ct_loader, **kwargs):
        super().__init__(ct_loader, **kwargs)
        self.knn        = KNN(n_neighbors = 5, metric = "euclidean")
        self.ct_loader  = ct_loader
        
    def update_knn(self):
        self.model.train(False)
        x_train = []
        y_train = []
        for batch in self.train_loader:
            x, y     = self.get_batch(batch)
            features = self.model(x)
            features = features.cpu().detach().numpy()
            x_train.append(features)
            y_train.append(y)
        x_train = np.concatenate(x_train, axis = 0)
        y_train = np.concatenate(y_train, axis = 0)
        self.knn.fit(x_train, y_train)
        
    def get_probabilities(self, set_loader):
        self.model.train(False)
        y_trues, y_probs = [], []
        for batch in set_loader:
            x, y_true   = self.get_batch(batch)
            features    = self.model(x)
            features    = features.cpu().detach().numpy()
            y_trues.append(y_true)
            y_probs.append( self.knn.predict_proba(features)[:,1] )
        y_true = np.concatenate(y_trues, axis = 0)
        y_prob = np.concatenate(y_probs, axis = 0)
        return y_true, y_prob
        
    def get_unsup_train_loader(self):
        unsup_train = self.ct_loader.load_train()
        unsup_train = torch.utils.data.DataLoader(torchio.SubjectsDataset(unsup_train),
                                    batch_size  = self.batch_size, 
                                    num_workers = self.num_workers, 
                                    pin_memory  = self.cuda)
        return unsup_train
    
    def train_epoch(self, epoch: int):
        self.trace(f"{self.model.get_name()} - epoch {epoch}/{self.epochs} --------------------------------")
        self.model.train(True)
        y_trues, y_probs = [], []
        for batch in self.get_unsup_train_loader():
            self.train_optimizer.zero_grad()  # reset gradients
            x, _      = self.get_batch(batch)
            features  = self.model(x)
            features1 = features[:self.batch_size//2]
            features2 = features[self.batch_size//2:]
            loss      = LOSS(features1, features2)
            loss.backward()                   # compute the loss and its gradients
            self.train_optimizer.step()       # adjust learning weights
        self.update_knn()
        return compute_metrics( *self.get_probabilities(self.train_loader) )
