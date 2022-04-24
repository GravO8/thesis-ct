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
    # torch.cuda.set_device(3)
    to_test = ["2.17.", "2.35.", "1.9.", "1.21.", "1.28."]
    for t in to_test:
        trainer = reload_trainer(f"MILNet-{t}")
        trainer.test(0)
        break
