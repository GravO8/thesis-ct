import torch
from aspects_side_comparator import ASPECTSSideComparator
from aspects_instance_model import ASPECTSInstanceModel

class ASPECTSInstanceClassifier(torch.nn.Module):
    def __init__(self, comparator: ASPECTSSideComparator, model: ASPECTSInstanceModel):
        super().__init__()
        self.comparator = comparator
        self.model      = model
        
    def __call__(self, instance):
        x = self.comparator(instance)
        y = self.model(x)
        return y
