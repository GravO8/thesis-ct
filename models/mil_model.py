# Adapted from:
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
import torch
import torch.nn as nn
from .mlp import MLP


class MILNet(nn.Module):
    def __init__(self, f: nn.Module, sigma: nn.Module, g: MLP):
        super(MILNet, self).__init__()
        # CNN input is NCHW
        self.f          = f
        self.sigma      = sigma
        self.g          = g
        self.return_features = self.g.return_features
    def forward(self, x):
        h       = self.f(x)         # instances encoding
        z, a    = self.sigma(h)     # MIL pooling obtains the bag encoding
        return self.g(z).squeeze()
    def to_dict(self):
        try:
            g_dict = self.g.to_dict()
        except:
            g_dict = self.g.__class__.__name__
        return {"type": "MIL", 
                "specs": {"f": self.f.to_dict(), 
                          "sigma": str(self.sigma),
                          "g": g_dict}}
        
        
class IlseAttention(nn.Module):
    def __init__(self, L: int = 128):
        super(IlseAttention, self).__init__()
        self.L          = L  # size of the MLP bottleneck
        self.attention  = nn.Sequential( nn.LazyLinear(L),
                                         nn.Tanh(),
                                         nn.Linear(L, 1))
    def forward(self, h):
        a = self.attention(h).T
        a = torch.nn.Softmax(dim = 1)(a)    # softmax over all the instance 
                                            # encodings, the attention weights normalized
        z = torch.mm(a, h)
        return z, a
    def __repr__(self):
        return f"IlseAttention({self.L})"


class IlseGatedAttention(nn.Module):
    def __init__(self, L: int = 128):
        super(IlseGatedAttention, self).__init__()
        self.L              = L  # size of the MLP bottleneck
        self.attention_V    = nn.Sequential(nn.LazyLinear(L),nn.Tanh())
        self.attention_U    = nn.Sequential(nn.LazyLinear(L),nn.Sigmoid())
        self.attention      = nn.Linear(L, 1)
    def forward(self, h):
        a_V = self.attention_V(h)
        a_U = self.attention_U(h)
        a   = self.attention(a_V * a_U).T
        a   = torch.nn.Softmax(dim = 1)(a)  # softmax over all the instance 
                                            # encodings, the attention weights normalized
        z = torch.mm(a, h)
        return z, a
    def __repr__(self):
        return f"IlseGatedAttention({self.L})"
        
        
class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
    def forward(self, h):
        z = h.mean(dim = 0)
        return z, None
    def __repr__(self):
        return "mean"


class Max(nn.Module):
    def __init__(self):
        super(Max, self).__init__()
    def forward(self, h):
        z = h.max(dim = 0).values
        return z, None
    def __repr__(self):
        return "max"
