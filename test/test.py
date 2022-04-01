import sys, torch
sys.path.append("..")
from utils.reload import Reload

def reload_trainer(dir: str, load_weights = True):
    '''
    TODO
    '''
    reload      = Reload(dir = dir)
    model       = reload.load_model(load_weights = load_weights)
    trainer     = reload.load_trainer()
    model_name  = reload.get_model_name()
    trainer.set_model(model, model_name)
    return trainer

if __name__ == "__main__":
    # torch.cuda.set_device(2)
    trainer = reload_trainer("SiameseNet-8.31.")
    # trainer.save_encodings()
