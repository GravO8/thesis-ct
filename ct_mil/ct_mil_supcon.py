import sys, torch, torchio, os, json
sys.path.append("..")
from utils.ct_loader_torchio import CTLoader
from models.cnn_2d_encoder import CNN2DEncoder
from timm.models.layers import GroupNorm
from models.mil_model import MILNet, IlseAttention, IlseGatedAttention, Mean, Max
from utils.trainer import MILTrainer
from utils.losses import SupConLoss
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
    else:
        NUM_WORKERS     = 8
        DATA_DIR        = "/media/avcstorage/gravo"
    HAS_BOTH_SCAN_TYPES     = False
    BALANCE_TEST_SET        = True
    BALANCE_TRAIN_SET       = True
    PATIENCE        = 40
    EPOCHS          = 300
    BATCH_SIZE      = 16
    CT_TYPE         = "NCCT"
    ct_loader       = CTLoader("table_data.csv", CT_TYPE, 
                            has_both_scan_types     = HAS_BOTH_SCAN_TYPES,
                            random_seed             = 0,
                            balance_test_set        = BALANCE_TEST_SET,
                            balance_train_set       = BALANCE_TRAIN_SET,
                            validation_size         = 0.1,
                            data_dir                = DATA_DIR,
                            target                  = "visible")
    loss_fn        = SupConLoss()
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
    VERSION        = "resnet18"
    i              = 4
    START          = 5
    skip           = True
    
    trainer.single(train_size = .8)
    for lr in (0.001, 0.0005, 0.0001):
        for weight_decay in (0.01, 0.001, 0.0001):
            for sigma in (IlseAttention(L = 128), IlseGatedAttention(L = 128)):
                i += 1
                if i == START:
                    skip = False
                if skip:
                    continue
                model_name  = MODEL_NAME.format(i)
                g           = torch.nn.Identity()
                g.__dict__["return_features"] = True
                model       = MILNet(f = CNN2DEncoder(cnn_name = VERSION,
                                                # drop_block_rate = d1, 
                                                # drop_rate = d1,
                                                # normalization = GroupNorm,
                                                pretrained = True,
                                                freeze = True,
                                                in_channels = 1),
                                    sigma = sigma,
                                    g = g)
                optimizer_args["lr"]            = lr
                optimizer_args["weight_decay"]  = weight_decay
                trainer.set_optimizer_args(optimizer_args)
                trainer.set_model(model, model_name)
                trainer.train()
                trainer.save_encodings()
