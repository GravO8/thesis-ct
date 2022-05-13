import torch, torchio
from ..model import Model


class Trainer:
    def __init__(self, ct_loader, batch_size: int = 32, epochs: int = 300, 
        patience: int = 2000, n_experiments: int = 3):
        if torch.cuda.is_available():
            self.cuda           = True
            self.num_workers    = 8
            self.trace_fn       = "log"
        else:
            self.cuda           = False
            self.num_workers    = 0
            self.trace_fn       = "print"
        self.set_loaders(ct_loader)

    def set_loaders(self, ct_loader):
        '''
        TODO
        '''
        train, validation, test = ct_loader.load_dataset()
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
                                            
    def train(self, model: Model):
        self.train_optimizer    = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self.writer             = SummaryWriter(self.model_name)
        early_stopping          = EarlyStopping(patience    = self.patience, 
                                                verbose     = True, 
                                                path        = self.weights,
                                                trace_func  = self.trace,
                                                delta       = .000001)
    
    
