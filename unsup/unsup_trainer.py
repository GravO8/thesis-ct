import sys
import numpy as np
sys.path.append("..")
from utils.trainer import Trainer, compute_metrics
from sklearn.neighbors import KNeighborsClassifier as KNN


class UnSupTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knn = KNN(n_neighbors = 5, metric = "euclidean")
        
    def set_loaders(self, ct_loader):
        '''
        TODO
        '''
        labeled_train, train, val, test = ct_loader.load_dataset()
        self.labeled_train = torch.utils.data.DataLoader(labeled_train, 
                                    batch_size  = self.batch_size, 
                                    num_workers = self.num_workers, 
                                    pin_memory  = self.cuda)
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
        
    def update_knn(self):
        self.model.train(False)
        x_train = []
        y_train = []
        for batch in self.labeled_train_loader:
            x, y = self.get_batch(batch)
            x_train.append( self.model(x).detach().numpy() )
            y_train.append( y )
        x_train = np.concatenate(x_train, axis = 0)
        y_train = np.concatenate(y_train, axis = 0)
        self.knn.fit(x_train, y_train)
        
    def get_probabilities(self, set_loader):
        self.model.train(False)
        y_trues, y_probs = [], []
        for batch in set_loader:
            x, y_true   = self.get_batch(batch)
            features    = self.model(x)
            features    = features.detach().numpy()
            y_trues.append(y_true)
            y_probs.append( self.knn.predict_proba(features)[:,1] )
        y_true = np.concatenate(y_trues, axis = 0)
        y_prob = np.concatenate(y_probs, axis = 0)
        return y_true, y_prob
    
    def train_epoch(self, epoch: int):
        self.trace(f"{self.model.get_name()} - epoch {epoch}/{self.epochs} --------------------------------")
        self.model.train(True)
        y_trues, y_probs = [], []
        for batch in self.train_loader:
            self.train_optimizer.zero_grad()  # reset gradients
            x, _     = self.get_batch(batch)
            features = self.model(x)
            loss     = LOSS(features)
            loss.backward()                   # compute the loss and its gradients
            self.train_optimizer.step()       # adjust learning weights
        self.update_knn()
        return compute_metrics( *self.get_probabilities(self.train_loader) )
