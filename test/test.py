import json, os, torch, torchio, sys
import sklearn.metrics as metrics
sys.path.append("..")
from utils.ct_loader_torchio import CT_loader
from models.siamese_model import SiameseNet
from models.mil_model import MIL_nn, Ilse_attention, Ilse_gated_attention, Mean, Max
from models.resnet3d import ResNet3D
from utils.trainer import SiameseTrainer, MILTrainer
from models.resnet import ResNet


def load_encoder(f):
    '''
    TODO
    '''
    try:
        s               = f.split("(")
        encoder_model   = s[0]
        if "resnet" in encoder_model:
            params  = s[1][:-1].split(",")
            encoder = ResNet( version = encoder_model, 
                            pretrained = params[0] == "True",
                            n_features = params[1])
    except:
        encoder = ResNet(   version = f["version"], 
                            pretrained = f["pretrained"] == "True",
                            n_features = f["n_features"],
                            freeze = f["freeze"] == "True")
    return encoder
    
    
def load_3d_encoder(encoder_info):
    '''
    TODO
    '''
    version         = encoder_info["version"]
    n_features      = encoder_info["n_features"]
    dropout         = encoder_info["dropout"]
    dropout         = None if dropout == "no" else float(dropout)
    normalization   = encoder_info["normalization"]
    return ResNet3D(version         = version, 
                    n_features      = n_features, 
                    dropout         = dropout,
                    normalization   = normalization)
                    
def load_siamese_mlp(mlp_info):
    '''
    TODO
    '''
    mlp_layers      = list(mlp_info["mlp_layers"])
    dropout         = mlp_info["dropout"]
    dropout         = None if dropout == "no" else float(dropout)
    return_features = mlp_info["return_features"] == "True"
    return {"mlp_layers": mlp_layers, "dropout": dropout, "return_features": return_features}

def load_sigma(sigma):
    '''
    TODO
    '''
    if "Ilse_attention" in sigma:
        bottleneck = int(sigma.split("(")[1][:-1])
        return Ilse_attention(L = bottleneck)
    elif "Ilse_gated_attention" in sigma:
        bottleneck = int(sigma.split("(")[1][:-1])
        return Ilse_gated_attention(L = bottleneck)
    elif sigma == "max":
        return Max()
    elif sigma == "mean":
        return Mean()
    else:
        assert False


def load_model(run_info, home):
    '''
    TODO
    '''
    model_info = run_info["model"]
    if model_info["type"] == "MIL":
        specs   = model_info["specs"]
        f       = load_encoder(specs["f"])
        sigma   = load_sigma(specs["sigma"])
        model   = MIL_nn(f = f, sigma = sigma)
    elif model_info["type"] == "SiameseNet":
        specs   = model_info["specs"]
        encoder = load_3d_encoder(specs["encoder"])
        mlp     = load_siamese_mlp(specs["mlp"])
        model   = SiameseNet(encoder = encoder, **mlp)
    else:
        assert False
    if not home:
        model.cuda()
    return model
        

def get_dir_contents(dir):
    '''
    TODO
    '''
    json_file, weights = None, None
    for file in os.listdir(dir):
        if file.endswith("json") and "summary" in file:
            json_file = f"{dir}/{file}"
        elif file.endswith("pt"):
            weights = f"{dir}/{file}"
    return json_file, weights


def load_trainer(run_info):
    '''
    TODO
    '''
    ct_loader_info = run_info["ct_loader"]
    ct_loader = CT_loader("gravo.csv", 
                            ct_type                 = ct_loader_info["ct_type"],
                            has_both_scan_types     = ct_loader_info["has_both_scan_types"] == "True",
                            binary_label            = ct_loader_info["binary_label"] == "True",
                            random_seed             = int(ct_loader_info["random_seed"]),
                            balance_test_set        = ct_loader_info["balance_test_set"] == "True",
                            balance_train_set       = ct_loader_info["balance_train_set"] == "True",
                            augment_factor          = float(ct_loader_info["augment_factor"]),
                            validation_size         = float(ct_loader_info["validation_size"]),
                            data_dir                = "../../../data/gravo" if home else "/media/avcstorage/gravo")
    model_type = run_info["model"]["type"]
    if model_type == "MIL":
        trainer = MIL_trainer(ct_loader, batch_size = 32, num_workers = 1 if home else 8)
    elif model_type == "SiameseNet":
        trainer = SiameseTrainer(ct_loader, batch_size = 32, num_workers = 1 if home else 8)
    else:
        assert False, f"load_trainer: Unknown model type {model_type}"
    mode = ct_loader_info["mode"]
    if mode == "SINGLE":
        trainer.single(train_size = float(ct_loader_info["train_size"]))
    elif mode == "KFOLD":
        k       = int(ct_loader_info["k"])
        fold    = int(ct_loader_info["fold"])
        trainer.k_fold(k = k)
        for _ in range(fold-1): # -1 because trainer.k_fold already loads the first fold
            trainer.next_fold()
    else:
        assert False, f"load_trainer: Unknown ct loader mode {mode}"
    return trainer
    
    
def test(dir_name: str):
    '''
    TODO
    '''
    json_file, weights_file = get_dir_contents(dir_name)
    with open(json_file, "r") as f:
        run_info = json.load(f)
    model = load_model(run_info, home)
    if home:
        model.load_state_dict( torch.load(weights_file, map_location = torch.device("cpu")) )
    else:
        model.load_state_dict( torch.load(weights_file) )
    trainer     = load_trainer(run_info)
    model_name  = dir_name.split("-fold")[0] if "fold" in dir_name else dir_name
    trainer.test(model, model_name)


home = not torch.cuda.is_available()
if __name__ == "__main__":
    test("SiameseNet-3.34.-fold1of5")
    test("SiameseNet-2.37.-fold1of5")
    test("SiameseNet-3.44.-fold1of5")
    # test("SiameseNet-3.1.-fold1of5")
