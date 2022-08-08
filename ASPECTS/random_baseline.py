import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aspects_mil_loader import ASPECTSMILLoader, REGIONS

N           = 10  # number of classes
L           = 43+18   # instance size
# L           = 22   # instance size


def load_set(set_name: str, tens: int = 10000):
    '''
    tens, the number of patients with an aspects of 10
    '''
    x, y = [], []
    odd, even = range(0,N*2,2), range(1,N*2,2)
    for x_sample, gt in loader.get_set(set_name):
        if gt == 10:
            tens -= 1
            if tens < 0: continue
        y.append(gt < 10)
        x.append( torch.abs(x_sample[odd]-x_sample[even]).numpy() )
    x = torch.Tensor(np.array(x))
    y = torch.Tensor(y).view(-1,1)
    return x, y


dirname = "../../../data/gravo"
# dirname = "/media/avcstorage/gravo/"
loader = ASPECTSMILLoader("ncct_radiomic_features.csv", "all", 
        normalize = True, dirname = dirname, set_col = "instance_aspects_set",
        feature_selection = L < 43)
_, y = load_set("test", tens = 10000)

def get_random_prediction(y):
    out = []
    y = int(y.numpy())
    for i in range(y):
        out.append(np.random.randint(5001,10000)/10000)
    for i in range(10-y):
        out.append(np.random.randint(0,4999)/10000)
    np.random.shuffle(out)
    i = 0
    sapo = {}
    for r in REGIONS:
        sapo[REGIONS[r]] = out[i]
        i += 1
    return sapo
    
def instance_level_performance(loader):
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = [], []
    for x, y in loader.get_set("test"):
        x_test.append(x)
        y_test.append(y)
    y_pred, y_true, positive = {REGIONS[r]: [] for r in REGIONS}, {REGIONS[r]: [] for r in REGIONS}, {REGIONS[r]: 0 for r in REGIONS}
    for i in range(len(x_test)):
        pred = get_random_prediction(y_test[i])
        for r in REGIONS:
            r = REGIONS[r]
            y_pred[r].append( int(np.round(pred[r])) )
            y_true[r].append( 0 if y_test[i]==10 else y_instance[i][r] )
            positive[r] += y_true[r][-1]
    out = {}
    metrics = {"accur": accuracy_score, "precis": precision_score, "recall": recall_score}
    for metric in metrics:
        out[metric] = {}
        for r in REGIONS:
            r = REGIONS[r]
            out[metric][r] = metrics[metric](y_true = y_true[r], y_pred = y_pred[r])
    return out, positive
    
def print_instance_level_performance(loader):
    performance, positive = instance_level_performance(loader)
    print("metric", end = "\t")
    for r in REGIONS: print(f"{REGIONS[r]} ({positive[REGIONS[r]]})", end = "\t")
    for metric in performance:
        print()
        print(metric, end = "\t")
        for r in performance[metric]:
            print(f"{performance[metric][r]*100:.2f}", end = "\t")
    print()

def instance_level_f1(loader):
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = [], []
    for x, y in loader.get_set("test"):
        x_test.append(x)
        y_test.append(y)
    y_pred, y_true = [], []
    for i in range(len(x_test)):
        pred = get_random_prediction(y_test[i])
        for r in REGIONS:
            r = REGIONS[r]
            y_pred.append( int(np.round(pred[r])) )
            y_true.append( 0 if y_test[i]==10 else y_instance[i][r] )
    return f1_score(y_true = y_true, y_pred = y_pred)
    
print(instance_level_f1(loader))
print_instance_level_performance(loader)
    
