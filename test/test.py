import sys
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
    test("SiameseNet-3.34.-fold1of5")
    test("SiameseNet-2.37.-fold1of5")
    test("SiameseNet-3.44.-fold1of5")
    # test("SiameseNet-3.1.-fold1of5")
