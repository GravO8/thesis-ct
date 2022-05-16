import torch


class Encoder(torch.nn.Module):
    def __init__(self, model_name: str, encoder: torch.nn.Module, out_features: int, 
    global_pool: str = None, dim: int = None):
        torch.nn.Module.__init__(self)
        self.encoder      = encoder
        self.model_name   = model_name
        self.out_features = out_features
        self.global_pool  = global_pool
        self.dim          = dim
        if self.global_pool is None:
            self.pooling  = torch.nn.Identity()
        else:
            assert self.dim in (2,3), "Encoder.__init__: specify dim when setting 'global_pool' (valid dims: {2,3})."
            assert self.global_pool in ("gap", "gmp"), "Encoder.__init__: valid values for 'global_pool': {'gap','gmp'}."
            if self.global_pool == "gap":   # global average pooling
                self.pooling = eval(f"torch.nn.AdaptiveAvgPool{self.dim}d")(1)
            elif self.global_pool == "gmp": # global max pooling
                self.pooling = eval(f"torch.nn.AdaptiveMaxPool{self.dim}d")(1)
    def get_name(self):
        return f"{self.model_name}_{'features' if self.global_pool is None else self.global_pool}_{self.dim}D"
    def forward(self, x):
        x = self.encoder(x)
        x = self.pooling(x).squeeze()
        assert self.out_features == x.shape[1], f"Encoder.forward: expected {self.out_features} out features, got {x.shape[1]}."
        return x


class SiameseEncoderMerger:
    def __init__(self, fn_name: str, fn):
        self.fn_name = fn_name
        self.fn      = fn
    def get_name(self):
        return self.fn_name
    def __call__(self, x1, x2):
        return self.fn(x1, x2)
        

class SiameseEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, merge_encodings: SiameseEncoderMerger, 
    merged_encoder: Encoder):
        torch.nn.Module.__init__(self)
        self.encoder         = encoder
        self.merge_encodings = merge_encodings
        self.merged_encoder  = merged_encoder
        if self.merged_encoder.global_pool is None:
            assert self.encoder.global_pool is not None, "SiameseEncoder.__init__: either the 'encoder' or the 'merged_encoder' must apply global pooling."
        else:
            assert self.encoder.global_pool is None, "SiameseEncoder.__init__: global pooling can't be applied by both the 'encoder' and the 'merged_encoder'."
    def get_name(self):
        return f"{self.encoder.get_name()}_{self.merge_encodings.get_name()}_{self.merged_encoder.get_name()}"
    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x  = self.merge_encodings(x1, x2)
        x  = self.merged_encoder(x) 
        assert self.merged_encoder.out_features == x.shape[1], f"SiameseEncoder.forward: expected {self.merged_encoder.out_features} out features, got {x.shape[1]}."
        return x
