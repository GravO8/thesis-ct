import json, os, torch, sys
import sklearn.metrics as metrics
sys.path.append("..")
from utils.reload import get_dir_contents, load_model, load_trainer
    
    
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
    trainer     = load_trainer(run_info, home)
    model_name  = dir_name.split("-fold")[0] if "fold" in dir_name else dir_name
    trainer.test(model, model_name)


home = not torch.cuda.is_available()
if __name__ == "__main__":
    test("SiameseNet-3.34.-fold1of5")
    test("SiameseNet-2.37.-fold1of5")
    test("SiameseNet-3.44.-fold1of5")
    # test("SiameseNet-3.1.-fold1of5")
