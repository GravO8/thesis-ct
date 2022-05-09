import sys, torch, torchio, os, json
sys.path.append("..")
from utils.ct_loader_torchio import TargetTransform, binary_mrs
from utils.ct_loader_2d import AxialLoader
from timm.models.layers import GroupNorm
from models.encoder_2d import Encoder2D
from utils.trainer import CNNTrainer2D
from models.mlp import MLP

# To check the number of GPUs and their usage, use:
# nvidia-smi
# To check processes running, use:
# top 
# htop
# ps -aux | grep gro


if __name__ == "__main__":
    home        = not torch.cuda.is_available()
    if home:
        NUM_WORKERS     = 0
        DATA_DIR        = "../../../data/gravo"
    else:
        torch.cuda.set_device(1)
        NUM_WORKERS     = 8
        DATA_DIR        = "/media/avcstorage/gravo"
    
    HAS_BOTH_SCAN_TYPES     = False
    BALANCE_TEST_SET        = True
    BALANCE_TRAIN_SET       = True
    PATIENCE        = 40
    EPOCHS          = 300
    BATCH_SIZE      = 32
    CT_TYPE         = "NCCT"
    CSV_FILENAME    = "table_data.csv"
    scan_ids        = [1303781, 2520986, 2605128, 2503602, 1911947]
    scan_slices     = [35, 33, 34, 36, 39]
    ct_loader       = AxialLoader(CSV_FILENAME, CT_TYPE, 
                            scan_ids,
                            scan_slices,
                            slice_interval          = 2,
                            pad                     = 224,
                            has_both_scan_types     = HAS_BOTH_SCAN_TYPES,
                            random_seed             = 0,
                            balance_test_set        = BALANCE_TEST_SET,
                            balance_train_set       = BALANCE_TRAIN_SET,
                            validation_size         = 0.1,
                            data_dir                = DATA_DIR,
                            target                  = "rankin-23",
                            target_transform        = TargetTransform("binary_mrs", binary_mrs),
                            transforms              = ["RandomFlip", "RandomNoise", "RandomElasticDeformation", "RandomAffine"])
    loss_fn         = torch.nn.BCELoss(reduction = "mean")
    optimizer       = torch.optim.Adam
    optimizer_args  = {"betas": (0.9, 0.999)}
    trainer         = CNNTrainer2D(ct_loader, 
                                optimizer   = optimizer, 
                                loss_fn     = loss_fn, 
                                trace_fn    = "print" if home else "log",
                                batch_size  = BATCH_SIZE,
                                num_workers = NUM_WORKERS,
                                epochs      = EPOCHS,
                                patience    = PATIENCE)
    MODEL_NAME      = "2DViT-1.{}."
    VERSION         = "vit_base_patch16_224"
    START           = 1
    i               = 0
    # skip            = True
    
    trainer.single(train_size = .8)
    trainer.assert_datasets()
    for lr in (0.001, 0.0005, 0.0001):
        for weight_decay in (.1, 0.01, 0.001, 0.0001):
            for freeze in (True, False):
                i += 1
                model_name  = MODEL_NAME.format(i)
                model       = Encoder2D(encoder_name    = VERSION, 
                                        n_features      = 1,
                                        pretrained      = True,
                                        freeze          = freeze)
                optimizer_args["lr"]            = lr
                optimizer_args["weight_decay"]  = weight_decay
                trainer.set_optimizer_args(optimizer_args)
                trainer.set_model(model, model_name)
                trainer.train()
