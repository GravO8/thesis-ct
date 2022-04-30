import sys, torch, torchio, os, json
sys.path.append("..")
from utils.ct_loader_torchio import CTLoader, TargetTransform, binary_mrs
from models.cnn_2d_encoder import CNN2DEncoder
from timm.models.layers import GroupNorm
from models.resnet3d import ResNet3D
from utils.trainer import CNNTrainer
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
    PATIENCE        = 2000
    EPOCHS          = 75
    BATCH_SIZE      = 16
    CT_TYPE         = "NCCT"
    CSV_FILENAME    = "table_data.csv"
    ct_loader       = CTLoader(CSV_FILENAME, CT_TYPE, 
                            has_both_scan_types     = HAS_BOTH_SCAN_TYPES,
                            random_seed             = 0,
                            balance_test_set        = BALANCE_TEST_SET,
                            balance_train_set       = BALANCE_TRAIN_SET,
                            validation_size         = 0.1,
                            data_dir                = DATA_DIR,
                            target                  = "rankin-23",
                            target_transform        = TargetTransform("binary_mrs", binary_mrs),
                            transforms              = ["RandomFlip", "RandomNoise", "RandomElasticDeformation", "RandomAffine"])
    loss_fn        = torch.nn.BCELoss(reduction = "mean")
    optimizer      = torch.optim.Adam
    optimizer_args = {"betas": (0.9, 0.999)}
    trainer        = CNNTrainer(ct_loader, 
                                optimizer   = optimizer, 
                                loss_fn     = loss_fn, 
                                trace_fn    = "print" if home else "log",
                                batch_size  = BATCH_SIZE,
                                num_workers = NUM_WORKERS,
                                epochs      = EPOCHS,
                                patience    = PATIENCE)
    MODEL_NAME     = "3DCNN-1.{}."
    VERSION        = "resnet-18"
    START          = 1
    i              = START-1
    # skip           = True
    
    trainer.single(train_size = .8)
    trainer.assert_datasets()
    for lr in (0.001, 0.0005, 0.0001):
        for weight_decay in (0.01, 0.001, 0.0001):
            for d1,d2 in ((.0, .0), (.1, .1), (.5, .5), (.8, .8)):
                i += 1
                model_name  = MODEL_NAME.format(i)
                model       = ResNet3D( version = VERSION, 
                                        n_features = 1,
                                        drop_rate = d1, 
                                        drop_block_rate = d2,
                                        normalization = "group")
                optimizer_args["lr"]            = lr
                optimizer_args["weight_decay"]  = weight_decay
                trainer.set_optimizer_args(optimizer_args)
                trainer.set_model(model, model_name)
                trainer.train()
