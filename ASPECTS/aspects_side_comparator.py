import torch
from abc import ABC, abstractmethod

class ASPECTSSideComparator(ABC):
    def __call__(self, x1, x2):
        return self.forward(x1, x2)
    @abstractmethod
    def forward(self, x1, x2):
        pass
        
        
class ABSDiff(ASPECTSSideComparator):
    def forward(x1, x2):
        return torch.abs(x1 - x2)

class Identity(ASPECTSSideComparator):
    def forward(x1, x2):
        return x1, x2
