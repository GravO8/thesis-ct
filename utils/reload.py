import json, os, torch, sys
sys.path.append("..")
from utils.ct_loader_torchio import CT_loader
from models.siamese_model import SiameseNet
from models.mil_model import MIL_nn, Ilse_attention, Ilse_gated_attention, Mean, Max
from models.resnet3d import ResNet3D
from utils.trainer import SiameseTrainer, MILTrainer
from models.resnet import ResNet


def load_sigma(sigma: dict):
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


def load_3d_encoder(encoder_info: dict):
    '''
    TODO
    '''
    version         = encoder_info["version"]
    n_features      = encoder_info["n_features"]
    normalization   = encoder_info["normalization"]
    try:
        dropout     = None if dropout == "no" else float(dropout)
        return ResNet3D(version         = version, 
                        n_features      = n_features, 
                        dropout         = dropout,
                        normalization   = normalization)
    except:
        drop_rate               = encoder_info["drop_rate"]
        drop_block_rate         = float(encoder_info["drop_block_rate"])
        remove_first_maxpool    = float(encoder_info["remove_first_maxpool"])
        return ResNet3D(version                 = version,
                        n_features              = n_features,
                        drop_rate               = drop_rate,
                        drop_block_rate         = drop_block_rate,
                        normalization           = normalization,
                        remove_first_maxpool    = remove_first_maxpool)


def load_encoder(f: dict):
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


def load_siamese_mlp(mlp_info):
    '''
    TODO
    '''
    dropout         = mlp_info["dropout"]
    return_features = mlp_info["return_features"] == "True"
    try:
        mlp_layers  = list(mlp_info["mlp_layers"])
        dropout     = None if dropout == "no" else float(dropout)
        return {"mlp_layers": mlp_layers, "dropout": dropout, 
        "return_features": return_features}
    except:
        layers = mlp_info["layers_list"]
        activation = mlp_info["hidden_activation"]
        if activation == "GELU":
            activation = torch.nn.GELU()
        else:
            assert False, f"load_siamese_mlp: Unknown activation {activation}"
        return {"layers_list": mlp_layers, "dropout": dropout, 
        "return_features": return_features, "hidden_activation": activation}


def load_optimizer(optimizer_info: dict, optimizer_args: dict = {}):
    '''
    TODO
    '''
    if "Adam" in optimizer_info["name"]:
        optimizer = torch.optim.Adam
    else:
        assert False, f"load_optimizer: Unknown optimizer {optimizer_info['name']}"
    for arg in ["lr", "betas", "weight_decay", "eps"]:
        optimizer_args[arg] = optimizer_args[arg] if arg in optimizer_args else optimizer_info[arg]
    return optimizer, optimizer_args
    

def load_loss(loss_str):
    if "BCELoss" in loss_str:
        return torch.nn.BCELoss(reduction = "mean")
    else:
        assert False, f"load_loss: Unknown loss {loss_str}"


class Reload:
    def __init__(self, dir):
        '''
        TODO
        '''
        self.dir    = dir
        self.cuda   = torch.cuda.is_available()
        self.load_run_info()
        
    def get_model_name(self):
        return self.dir.split("-fold")[0] if "fold" in self.dir else self.dir
        
    def load_run_info(self):
        '''
        TODO
        '''
        with open(f"{self.dir}/summary.json", "r") as f:
            self.run_info = json.load(f)
            
    def get_weights(self):
        '''
        TODO
        '''
        return [f"{self.dir}/{filename}" for filename in os.listdir(self.dir) if filename.endswith(".pt")]
            
    def load_model(self, load_weights = True):
        '''
        TODO
        '''
        model_info = self.run_info["model"]
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
        if self.cuda:
            model.cuda()
        if load_weights:
            weights = self.get_weights()
            assert len(weights) > 0, "Reload.load_model: no weights found. Use load_weights=False or train the model first"
            if len(weights) > 1:
                print(f"Reload.load_model: more than one weights file found. Using {weights[0]}.")
            if self.cuda:
                model.load_state_dict( torch.load(weights[0], map_location = torch.device("cpu")) )
            else:
                model.load_state_dict( torch.load(weights[0]) )
        return model
        
    def load_ct_loader(self):
        '''
        TODO
        '''
        ct_loader_info = self.run_info["ct_loader"]
        ct_loader = CT_loader("gravo.csv", 
                                ct_type                 = ct_loader_info["ct_type"],
                                has_both_scan_types     = ct_loader_info["has_both_scan_types"] == "True",
                                binary_label            = ct_loader_info["binary_label"] == "True",
                                random_seed             = int(ct_loader_info["random_seed"]),
                                balance_test_set        = ct_loader_info["balance_test_set"] == "True",
                                balance_train_set       = ct_loader_info["balance_train_set"] == "True",
                                augment_factor          = float(ct_loader_info["augment_factor"]),
                                validation_size         = float(ct_loader_info["validation_size"]),
                                data_dir                = "/media/avcstorage/gravo" if self.cuda else "../../../data/gravo")
        return ct_loader, ct_loader_info
        
    def init_trainer(self, ct_loader, epochs: int, patience: int):
        '''
        TODO
        '''
        optimizer, optimizer_args = load_optimizer( self.run_info["optimizer"] )
        model_type   = self.run_info["model"]["type"]
        trainer_args = {"optimizer":        optimizer,
                        "optimizer_args":   optimizer_args,
                        "trace_fn":         "log" if self.cuda else "print",
                        "num_workers":      8 if self.cuda else 1,
                        "batch_size":       self.run_info["train"]["batch_size"],
                        "loss_fn":          load_loss(self.run_info["loss_fn"]),
                        "epochs":           epochs,
                        "patience":         patience}
        if model_type == "MIL":
            trainer = MIL_trainer(ct_loader, **trainer_args)
        elif model_type == "SiameseNet":
            trainer = SiameseTrainer(ct_loader, **trainer_args)
        else:
            assert False, f"Reload.load_trainer: Unknown model type {model_type}"
        return trainer

    def load_trainer(self, epochs: int = 0, patience: int = 0, optimizer_args: dict = {}):
        '''
        TODO
        '''
        ct_loader, ct_loader_info   = self.load_ct_loader()
        trainer                     = self.init_trainer(ct_loader, epochs, patience)
        mode                        = ct_loader_info["mode"]
        if mode == "SINGLE":
            trainer.single(train_size = float(ct_loader_info["train_size"]))
        elif mode == "KFOLD":
            k       = int(ct_loader_info["k"])
            fold    = int(ct_loader_info["fold"])
            trainer.k_fold(k = k)
            for _ in range(fold-1): # -1 because trainer.k_fold already loads the first fold
                trainer.next_fold()
        else:
            assert False, f"Reload.load_trainer: Unknown ct loader mode {mode}"
        return trainer
