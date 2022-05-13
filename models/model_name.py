def to_model_family(i: int):
    return ["baseline", "siamese", "MIL", "axial"][i]
    
def baseline_experiment(i: int):
    return ["3DCNN"][i]
    
def siamese_experiment(i: int):
    return ["before", "after", "tangled"][i]
    
def MIL_experiment(i: int):
    return ["axial-max", "axial-max-pretrained", "axial-max-frozen", "axial-mean", 
    "axial-attention", "cubes-max", "cubes-mean", "cubes-attention"][i]
    
def axial_experiment(i: int):
    return [""]
