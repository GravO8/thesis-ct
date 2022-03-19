import sys, torch, torchio, os, json
sys.path.append("..")
from utils.ct_loader_torchio import CT_loader
from models.mil_model import MIL_nn, Ilse_attention, Ilse_gated_attention, Mean, Max
from utils.trainer import MILTrainer
from models.resnet import ResNet

# To check the number of GPUs and their usage, use:
# nvidia-smi
# To check processes running, use:
# top 
# htop
        
        
class NLL:
    def __init__(self):
        pass
    def __call__(self, Y_prob, Y):
        return -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    def __repr__(self):
        return "negative log bernoulli"
        
        
def to_try():
    return [dir for dir in os.listdir() if "MIL" in dir]


if __name__ == "__main__":
    home = not torch.cuda.is_available()
    if home:
        NUM_WORKERS     = 0
        DATA_DIR        = "../../data/gravo"
        HAS_BOTH_SCAN_TYPES     = True
        BALANCE_TEST_SET        = False
        BALANCE_TRAIN_SET       = False
        AUGMENT_FACTOR          = 1
    else:
        NUM_WORKERS     = 8
        DATA_DIR        = "/media/avcstorage/gravo"
        HAS_BOTH_SCAN_TYPES     = False
        BALANCE_TEST_SET        = True
        BALANCE_TRAIN_SET       = True
        AUGMENT_FACTOR          = 5
    
    PATIENCE        = 2000
    EPOCHS          = 75
    ct_loader       = CT_loader("gravo.csv", "NCCT", 
                            has_both_scan_types     = HAS_BOTH_SCAN_TYPES,
                            binary_label            = True,
                            random_seed             = 0,
                            balance_test_set        = BALANCE_TEST_SET,
                            balance_train_set       = BALANCE_TRAIN_SET,
                            augment_factor          = AUGMENT_FACTOR,
                            validation_size         = 0.1,
                            data_dir                = DATA_DIR)
    loss_fn         = NLL()
    for model_name in to_try():
        json_file, _    = get_dir_contents(model_name)
        with open(json_file, "r") as f:
            run_info = json.load(f)
        model           = load_model(run_info, home)
        trainer         = MILTrainer(model_name, 
                                ct_loader, model, 
                                trace_fn = "print" if home else "log")
        optimizer       = torch.optim.Adam(trainer.model.parameters(), lr = 0.0005, betas = (0.9, 0.999), weight_decay = 10e-5)
        trainer.train(optimizer, loss_fn, 1, NUM_WORKERS, patience = PATIENCE, epochs = EPOCHS)
