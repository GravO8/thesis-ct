import torch, copy
from aspects_instance_classifier import ASPECTSInstanceClassifier

class ASPECTSBagClassifier(torch.nn.Module):
    def __init__(self, instance_classifier: ASPECTSInstanceClassifier, 
    share_weights: bool = False):
        super().__init__()
        self.share_weights = share_weights
        if not self.share_weights:
            instance_classifier = [copy.deepcopy(instance_classifier) for _ in range(10)]
        self.instance_classifer = instance_classifier
        
    def __call__(self, bag):
        if self.share_weights:
            x = self.instance_classifer(bag)
        else:
            x = torch.stack([self.instance_classifer[i](bag[i]) for i in range(10)], dim = 0)
        assert x.shape() == (10,1)
        y = x.sum().long()
        return torch.nn.functional.one_hot(t, num_classes = 11)
        
