import sys, torch
sys.path.append("..")
from utils.reload import Reload
    
    
def test(dir: str):
    '''
    TODO
    '''
    reload      = Reload(dir = dir)
    model       = reload.load_model(load_weights = True)
    trainer     = reload.load_trainer()
    model_name  = reload.get_model_name()
    trainer.test(model, model_name)


if __name__ == "__main__":
    torch.cuda.set_device(2)
    test("SiameseNet-8.12.")
    test("SiameseNet-8.20.")
    test("SiameseNet-8.31.")
