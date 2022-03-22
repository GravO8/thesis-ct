# Adapted from:
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
import torch
import torch.nn as nn
from .mlp import MLP


class MIL_nn(nn.Module):
    def __init__(self, f: nn.Module, sigma: nn.Module, g: MLP):
        super(MIL_nn, self).__init__()
        # CNN input is NCHW
        self.f          = f
        self.sigma      = sigma
        self.g          = g
    def forward(self, x):
        h       = self.f(x)         # instances encoding
        z, a    = self.sigma(h)     # MIL pooling obtains the bag encoding
        out     = self.g(z)
        if not self.g.return_features:
            y_prob  = out.squeeze()
            y_pred  = torch.ge(y_prob, 0.5).float()
            return y_prob, y_pred
        return out
    def to_dict(self):
        return {"type": "MIL", 
                "specs": {"f": self.f.to_dict(), 
                          "sigma": str(self.sigma),
                          "g": self.g.to_dict()}}
        
        
class Ilse_attention(nn.Module):
    def __init__(self, L: int = 128):
        super(Ilse_attention, self).__init__()
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
        return f"Ilse_attention({self.L})"


class Ilse_gated_attention(nn.Module):
    def __init__(self, L: int = 128):
        super(Ilse_gated_attention, self).__init__()
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
        return f"Ilse_gated_attention({self.L})"
        
        
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
