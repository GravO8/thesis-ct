import torch
from abc import ABC, abstractmethod

class ASPECTSSideComparator(ABC):
    def __call__(self, instance):
        x1 = instance[[0,2,4,6,8,10,12,14,16,18]]
        x2 = instance[[1,3,5,7,9,11,13,15,17,19]]
        return self.forward(x1, x2)
    @abstractmethod
    def forward(self, x1, x2):
        pass
        
        
class ABSDiff(ASPECTSSideComparator):
    def forward(x1, x2):
        # (10,N)
        return torch.abs(x1 - x2)

class Identity(ASPECTSSideComparator):
    def forward(x1, x2):
        # (10,2,N)
        return [[x1[i],x2[i]] for i in range(10)]
