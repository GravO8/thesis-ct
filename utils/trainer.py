import torch, torchio
from torch.utils.tensorboard import SummaryWriter

LR          = 0.0005 # learning rate
WD          = 0.0001 # weight decay
LOSS        = torch.nn.BCELoss(reduction = "mean")
OPTIMIZER   = torch.optim.Adam


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
            val_metrics     = self.get_metrics(self.validation_loader)
            test_metrics    = self.get_metrics(self.tset_loader)
            self.save_metrics(train_metrics, val_metrics, test_metrics, verbose = True)
            self.save_weights(val_metrics["f1_score"])
            
    def train_epoch(self):
        self.model.train(True)
        loss_cmul   = 0
        count       = 0
        ys, y_probs = [], []
        for subjects in self.train_loader:
            self.train_optimizer.zero_grad()  # reset gradients
            x, y    = self.get_batch(subjects)
            y_prob  = self.model(x)
            y_prob  = y_prob.squeeze().clamp(min = 1e-5, max = 1.-1e-5).cpu()
            loss    = LOSS(y_prob, y)
            loss.backward()                   # compute the loss and its gradients
            self.train_optimizer.step()       # adjust learning weights
            ys.append(y)
            y_probs.append(y_prob.detach().numpy())
        y       = np.concatenate(ys, axis = 0)
        y_prob  = np.concatenate(y_probs, axis = 0)
        return self.compute_metrics(y, y_prob)
            
    def get_batch(self, subjects):
        scans   = subjects["ct"][torchio.DATA]
        y       = subjects["target"].float()
        if self.cuda:
            scans = scans.cuda()
        return scans, y
        
    def compute_metrics(y, y_prob):
        y_pred = 
        
    
    
if __name__ == '__main__':
    from ct_loader import CTLoader
    ct_loader = CTLoader(data_dir = "../../../data/gravo")
    trainer = Trainer(ct_loader)
