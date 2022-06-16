import os, sys, torch, torchio, contextlib
import torch.nn.functional as F
sys.path.append("..")
from torch.utils.tensorboard import SummaryWriter
from utils.half_ct_loader import HalfCTLoader
from utils.trainer import Trainer, PERFORMANCE
from torchsummary import summary
from utils.logger import Logger


class LossTracker:
    REC = "rec"
    KLD = "kld"
    VAE = "vae"
    def __init__(self):
        self.loss = {LossTracker.REC: 0, LossTracker.KLD: 0, LossTracker.VAE: 0}
        self.count = 0
    def update(self, rec, kld, vae, c):
        self.loss[LossTracker.REC] += float(rec)
        self.loss[LossTracker.KLD] += float(kld)
        self.loss[LossTracker.VAE] += float(vae)
        self.count += int(c)
    def average(self):
        out = {}
        for l in self.loss:
            out[l] = self.loss[l]/self.count
        return out

LR          = 0.001
WD          = 0.0
OPTIMIZER   = torch.optim.Adam


def reconstruction_loss(x, x_recon, distribution):
    # copied from https://github.com/1Konny/Beta-VAE/blob/977a1ece88e190dd8a556b4e2efb665f759d0772/solver.py#L21
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    # copied from https://github.com/1Konny/Beta-VAE/blob/977a1ece88e190dd8a556b4e2efb665f759d0772/solver.py#L36
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class VAETrainer(Trainer):
    def __init__(self, ct_loader, beta = 2, **kwargs):
        super().__init__(ct_loader, **kwargs)
        self.half = isinstance(ct_loader, HalfCTLoader)
        if self.half:
            self.pad = ct_loader.pad is not None
        else:
            self.pad = False
        self.beta = beta
        
    def set_train_model(self, model):
        self.model = model.float()
        if self.cuda:
            print("Using cuda device")
            self.model.cuda()
        else:
            print("Using CPU device")
        model_name = self.model.get_name()
        if not os.path.isdir(model_name):
            os.system(f"mkdir {model_name}")
            with open(os.path.join(model_name, PERFORMANCE), "w") as f:
                f.write("model_name;run;best_epoch;set;recon_loss;kld;vae_loss\n")
            with open(os.path.join(model_name, "summary.txt"), "w") as f:
                f.write( str(self.model) )
                f.write("\n")
                with contextlib.redirect_stdout(f): # redirects print output to the summary.txt file
                    if self.half:
                        if self.pad:
                            summary(self.model, (1,64,128,128))
                        else:
                            summary(self.model, (1,46,109,91))
                    else:
                        summary(self.model, (1,91,109,91))
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

    def eval_batch(self, batch):
        x = batch["ct"][torchio.DATA].float() / 100
        if self.cuda:
            x = x.cuda()
        x_recon, mu, logvar = self.model(x)
        recon_loss          = reconstruction_loss(x, x_recon, "gaussian")
        total_kld, _, _     = kl_divergence(mu, logvar)
        beta_vae_loss       = recon_loss + self.beta*total_kld
        return recon_loss, total_kld, beta_vae_loss
        
    def evaluate(self, set_loader):
        self.model.train(False)
        loss = LossTracker()
        for batch in set_loader:
            loss.update(*self.eval_batch(batch), len(batch["patient_id"]))
        return loss.average()
        
    def train(self, model, lr = LR, weight_decay = WD, optimizer = OPTIMIZER):
        self.set_train_model(model)
        self.train_optimizer = optimizer(self.model.parameters(), 
                                        lr = lr,
                                        weight_decay = weight_decay)
        for epoch in range(self.epochs):
            train_loss   = self.train_epoch(epoch)
            val_loss     = self.evaluate(self.val_loader)
            test_loss    = self.evaluate(self.test_loader)
            self.save_loss(epoch, train_loss, val_loss, test_loss, verbose = True)
            self.save_weights(val_loss[LossTracker.VAE], epoch)
        self.record_performance()
        self.reset_model()
    
    def train_epoch(self, epoch: int):
        # adapted from https://github.com/1Konny/Beta-VAE/blob/977a1ece88e190dd8a556b4e2efb665f759d0772/solver.py#L159
        self.trace(f"{self.model.get_name()} - epoch {epoch}/{self.epochs} --------------------------------")
        self.model.train(True)
        loss = LossTracker()
        for batch in self.train_loader:
            self.train_optimizer.zero_grad()  # reset gradients
            recon_loss, kld, vae_loss = self.eval_batch(batch)
            vae_loss.backward()               # compute the loss and its gradients
            self.train_optimizer.step()       # adjust learning weights
            loss.update(recon_loss, kld, vae_loss, len(batch["patient_id"]))
        return loss.average()
        
    def tensorboard_loss(self, epoch: int, loss: dict):
        set_names    = ("train", "val", "test")
        metric_names = (LossTracker.REC, LossTracker.KLD, LossTracker.VAE)
        for set in loss:
            for metric in metric_names:
                self.writer.add_scalar(f"{metric}/{set}", loss[set][metric], epoch)
        self.writer.flush()
    
    def save_loss(self, epoch: int, train_loss: dict, val_loss: dict, 
        test_loss: dict, verbose: bool = True):
        loss = {"train": train_loss, "val": val_loss, "test": test_loss}
        self.tensorboard_loss(epoch, loss)
        if verbose:
            row = "{:<10}"*4
            self.trace(row.format("", LossTracker.REC, LossTracker.KLD, LossTracker.VAE))
            for set in loss:
                self.trace(row.format(set, round(loss[set][LossTracker.REC],2), 
                round(loss[set][LossTracker.KLD],2), round(loss[set][LossTracker.VAE],2)))
                
    def save_weights(self, vae_loss, epoch, verbose = True):
        if (self.best_score is None) or (vae_loss < self.best_score):
            if verbose:
                self.trace(f"Validation vae loss decreased ({self.best_score} --> {vae_loss}).  Saving model ...")
            self.best_score = vae_loss
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.weights_path)
                
    def record_performance(self):
        self.model.load_state_dict(torch.load(self.weights_path))
        model_name       = self.model.get_name()
        set_loaders      = {"train": self.train_loader, "val": self.val_loader, "test": self.test_loader}
        with open(os.path.join(model_name, PERFORMANCE), "a") as f:
            performance_row = f"{model_name};{self.run};{self.best_epoch}" + (";{}"*3) + "\n"
            for set in set_loaders:
                loss = self.evaluate( set_loaders[set] )
                f.write(performance_row.format(set, loss[LossTracker.REC], 
                loss[LossTracker.KLD], loss[LossTracker.VAE]))
