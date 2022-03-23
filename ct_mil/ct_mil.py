import sys, torch, torchio, os, json
sys.path.append("..")
from utils.ct_loader_torchio import CTLoader
from timm.models.layers import GroupNorm
from models.mil_model import MILNet, IlseAttention, IlseGatedAttention, Mean, Max
from utils.trainer import MILTrainer
from models.resnet import ResNet
from models.mlp import MLP

# To check the number of GPUs and their usage, use:
# nvidia-smi
# To check processes running, use:
# top 
# htop


if __name__ == "__main__":
    home        = not torch.cuda.is_available()
    if home:
        NUM_WORKERS     = 0
        DATA_DIR        = "../../../data/gravo"
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
    BATCH_SIZE      = 16
    CT_TYPE         = "NCCT"
    ct_loader       = CTLoader("gravo.csv", CT_TYPE, 
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
    optimizer_args = {"betas": (0.9, 0.999)}
    trainer        = MILTrainer(ct_loader, 
                                optimizer   = optimizer, 
                                loss_fn     = loss_fn, 
                                trace_fn    = "print" if home else "log",
                                batch_size  = BATCH_SIZE,
                                num_workers = NUM_WORKERS,
                                epochs      = EPOCHS,
                                patience    = PATIENCE)
    MODEL_NAME     = "MILNet-1.{}."
    VERSION        = "resnet34"
    i              = 0
    START          = 1
    skip           = True
    
    trainer.single(train_size = .8)
    for lr in (0.001, 0.0005, 0.0001):
        for weight_decay in (0.01, 0.001, 0.0001):
            for d1,d2 in ((.0, .0), (.1, .1), (.5, .5), (.8, .8)):
                for sigma in (IlseAttention(L = 128), Mean(), Max()):
                    i += 1
                    if i == START:
                        skip = False
                    if skip:
                        continue
                    model_name  = MODEL_NAME.format(i)
                    model       = MILNet(f = ResNet(drop_block_rate = d1, 
                                                    drop_rate = d1,
                                                    normalization = GroupNorm,
                                                    pretrained = False,
                                                    freeze = False,
                                                    in_channels = 1),
                                        sigma = sigma,
                                        g = MLP([256], 
                                                dropout = d2, 
                                                return_features = False, 
                                                n_out = 1))
                    optimizer_args["lr"]            = lr
                    optimizer_args["weight_decay"]  = weight_decay
                    trainer.set_optimizer_args(optimizer_args)
                    trainer.train(model, model_name)
