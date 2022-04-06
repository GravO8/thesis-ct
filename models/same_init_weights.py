import torch, json, os
from abc import ABC, abstractmethod

class SameInitWeights(ABC):
    def __init__(self):
        super(SameInitWeights, self).__init__()
        self.model_name = self.__class__.__name__
        self.set_model()
        self.init_weights()
        print("here")

    def init_weights(self):
        models      = [file for file in os.listdir("weights") if (file.endswith(".json") and file.startswith(f"{self.model_name}-"))]
        new_weights = True
        for model in models:
            if self.same_model(model):
                self.load_weights(model)
                new_weights = False
        if new_weights:
            new_model = f"weights/{self.model_name}-{len(models)}"
            torch.save(self.state_dict(), f"{new_model}.pt")
            with open(f"{new_model}.json", "w") as f:
                json.dump(self.to_dict(), f, indent = 4)
    
    def load_weights(self, model_name: str):
        weights = f"weights/{model_name[:-5]}.pt"
        if torch.cuda.is_available():
            self.load_state_dict( torch.load(weights) )
        else:
            self.load_state_dict( torch.load(weights, map_location = torch.device("cpu")) )
                
    def same_model(self, other_model: str) -> bool:
        with open(f"weights/{other_model}") as json_file:
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
