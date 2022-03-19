import sys, torch, torchio
sys.path.append("../CT-loader")
sys.path.append("../CT-MIL")
sys.path.append("../utils")
from ct_loader_torchio import CT_loader
from torch.utils.data import DataLoader
from siamese_model import SiameseNet
from resnet3d import ResNet3D
from trainer import Trainer
from losses import SupConLoss


class SiameseTrainer(Trainer):
    def evaluate_brain(self, subjects, verbose = False):
        scans       = subjects["ct"][torchio.DATA]
        if self.cuda:
            scans   = scans.cuda()
        msp         = scans.shape[2]//2             # midsagittal plane
        hemisphere1 = scans[:,:,:msp,:,:]           # shape = (B,C,x,y,z)
        hemisphere2 = scans[:,:,msp:,:,:].flip(2)   # B - batch; C - channels
        return self.model(hemisphere1, hemisphere2)

if __name__ == "__main__":
    home        = not torch.cuda.is_available()
    if home:
        NUM_WORKERS     = 0
        DATA_DIR        = "../../data/gravo"
        HAS_BOTH_SCAN_TYPES     = True
        BALANCE_TEST_SET        = not False ###############################################
        BALANCE_TRAIN_SET       = False
        AUGMENT_FACTOR          = 1
    else:
        NUM_WORKERS     = 8
        DATA_DIR        = "/media/avcstorage/gravo"
        HAS_BOTH_SCAN_TYPES     = False
        BALANCE_TEST_SET        = True
        BALANCE_TRAIN_SET       = True
        AUGMENT_FACTOR          = 5
    
    PATIENCE        = 40
    EPOCHS          = 300
    BATCH_SIZE      = 32
    ct_loader       = CT_loader("gravo.csv", "NCCT", 
                            has_both_scan_types     = HAS_BOTH_SCAN_TYPES,
                            binary_label            = True,
                            random_seed             = 0,
                            balance_test_set        = BALANCE_TEST_SET,
                            balance_train_set       = BALANCE_TRAIN_SET,
                            augment_factor          = AUGMENT_FACTOR,
                            validation_size         = 0.1,
                            data_dir                = DATA_DIR)
    # loss_fn        = SupConLoss()
    loss_fn        = torch.nn.BCELoss(reduction = "mean")
    optimizer      = torch.optim.Adam
    optimizer_args = {"lr": 0.0005, "betas": (0.9, 0.999), "weight_decay": 10e-5}
    trainer        = SiameseTrainer(ct_loader, optimizer, loss_fn, 
                                trace_fn    = "print" if home else "log",
                                batch_size  = BATCH_SIZE,
                                num_workers = NUM_WORKERS,
                                epochs      = EPOCHS,
                                patience    = PATIENCE)
    MODEL_NAME     = "SiameseNet-4.{}."
    VERSION        = "resnet50d"
    k              = 5
    trainer.k_fold(k)
    for fold in range(k):
        i = 1
        for lr in (.001, .0001, .00001):
            for weight_decay in (0.01, 0.001, 0.0001):
                for d1,d2 in ((.1, .5), (.1, .8), (.5, .5), (None, None)):
                    model_name                      = MODEL_NAME.format(i)
                    model                           = SiameseNet(encoder = ResNet3D(version = VERSION, 
                                                                                    dropout = d1,
                                                                                    normalization = "layer"), 
                                                                dropout = d2)
                                                                # return_features = True)
                    optimizer_args["lr"]            = lr
                    optimizer_args["weight_decay"]  = weight_decay
                    trainer.set_optimizer_args(optimizer_args)
                    trainer.train(model, model_name)
                    i += 1
        trainer.next_fold()
    
        
