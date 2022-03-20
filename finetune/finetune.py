import sys
sys.path.append("..")
from utils.reload import Reload

def finetune(dir: str):
    '''
    TODO
    '''
    reload      = Reload(dir = dir)
    model       = reload.load_model(load_weights = True)
    trainer     = reload.load_trainer(epochs = 100, patience = 100, {"lr": 0.00001})
    model_name  = reload.get_model_name()
    trainer.train(model, model_name)


if __name__ == "__main__":
    finetune("SiameseNet-3.44.-fold1of5")
