import torch, json, os
from abc import ABC, abstractmethod

class SameInitWeights(ABC):
    def __init__(self):
        super(SameInitWeights, self).__init__()
        self.model_name = self.__class__.__name__
        current_dir     = os.path.dirname(os.path.realpath(__file__))
        self.weight_dir = os.path.join(current_dir, "weights")
        self.set_model()
        self.init_weights()

    def init_weights(self):
        models      = [file for file in os.listdir(self.weight_dir) if (file.endswith(".json") and file.startswith(f"{self.model_name}-"))]
        new_weights = True
        for model in models:
            if self.same_model(model):
                self.load_weights(model)
                new_weights = False
                break
        if new_weights:
            new_model = os.path.join(self.weight_dir, f"{self.model_name}-{len(models)}")
            torch.save(self.state_dict(), f"{new_model}.pt")
            with open(f"{new_model}.json", "w") as f:
                json.dump(self.to_dict(), f, indent = 4)
    
    def load_weights(self, model_name: str):
        weights = os.path.join(self.weight_dir, f"{model_name[:-5]}.pt")
        if torch.cuda.is_available():
            self.load_state_dict( torch.load(weights) )
        else:
            self.load_state_dict( torch.load(weights, map_location = torch.device("cpu")) )
                
    def same_model(self, other_model: str) -> bool:
        with open(os.path.join(self.weight_dir, other_model)) as json_file:
            other_model = json.load(json_file)
        return self.equals(other_model)
        
    def equals(self, other_model: dict, cols: list = None) -> bool:
        self_model = self.to_dict()
        for col in (other_model if cols is None else cols):
            if self_model[col] != other_model[col]:
                return False
        return True
    
    @abstractmethod
    def set_model(self):
        '''
        TODO
        '''
        pass

    @abstractmethod
    def to_dict(self):
        pass
