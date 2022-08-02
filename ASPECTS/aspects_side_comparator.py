import torch
from abc import ABC, abstractmethod

class ASPECTSSideComparator(ABC):
    def __call__(self, instance):
        x1 = instance[range(0,len(instance),2)]
        x2 = instance[range(1,len(instance),2)]
        return self.forward(x1, x2)
    @abstractmethod
    def forward(self, x1, x2):
        pass
        
        
class ABSDiff(ASPECTSSideComparator):
    def forward(self, x1, x2):
        # (10,N)
        return torch.abs(x1 - x2)

class Identity(ASPECTSSideComparator):
    def forward(self, x1, x2):
        # (10,2,N)
        return [[x1[i],x2[i]] for i in range(10)]
