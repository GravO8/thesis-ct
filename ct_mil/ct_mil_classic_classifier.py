import sys, torch, torchio, os, json
sys.path.append("..")
from utils.ct_loader_torchio import CTLoader
from models.cnn_2d_encoder import CNN2DEncoder
from timm.models.layers import GroupNorm
from models.mil_model import MILNet, IlseAttention, IlseGatedAttention, Mean, Max
from utils.trainer import MILTrainer
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
        BALANCE_TEST_SET        = False
        BALANCE_TRAIN_SET       = False
    else:
        NUM_WORKERS     = 8
        DATA_DIR        = "/media/avcstorage/gravo"
        HAS_BOTH_SCAN_TYPES     = False
        BALANCE_TEST_SET        = True
        BALANCE_TRAIN_SET       = True
    
    PATIENCE        = -1
    EPOCHS          = -1
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
    loss_fn        = None
    optimizer      = None
    optimizer_args = None
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
    i              = 0
    START          = 1
    skip           = True
    
    # trainer.k_fold(k = 5)
    trainer.single(train_size = .8)
    trainer.assert_datasets()
    # for sigma in (Mean(), Max()):
    #     i += 1
    #     if i == START:
    #         skip = False
    #     if skip:
    #         continue
    #     model_name  = MODEL_NAME.format(i)
    #     model       = MILNet(f = CNN2DEncoder(version = VERSION,
    #                                     drop_block_rate = d1, 
    #                                     drop_rate = d1,
    #                                     pretrained = True,
    #                                     freeze = True,
    #                                     in_channels = 1),
    #                         sigma = sigma,
    #                         g = torch.nn.Identity())
    #     # normalization = GroupNorm,
    #     trainer.set_model(model, model_name)
    #     trainer.save_encodings()
