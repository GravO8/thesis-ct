import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aspects_mil_loader import ASPECTSMILLoader, REGIONS
from tqdm import tqdm

N           = 10  # number of classes
L           = 43
# +18   # instance size
# L           = 22   # instance size
LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
LR          = 0.001 # learning rate
WD          = 0.001
EPOCHS      = 1000
STEP_SIZE   = 400 # step size to update the LR
odd, even   = range(0,N*2,2), range(1,N*2,2)

def load_set(set_name: str, tens: int = 10000):
    '''
    tens, the number of patients with an aspects of 10
    '''
    x, y = [], []
    for x_sample, gt in loader.get_set(set_name):
        if gt == 10:
            tens -= 1
            if tens < 0: continue
        y.append(gt)
        x.append(torch.cat((x_sample[odd],x_sample[even]), dim = 1).numpy())
    x = torch.Tensor(np.array(x))
    y = torch.Tensor(y).view(-1,1)
    return x, y

dirname = "../../../data/gravo"
# dirname = "/media/avcstorage/gravo/"
loader = ASPECTSMILLoader("ncct_radiomic_features.csv", "radiomics", 
        normalize = True, dirname = dirname, set_col = "instance_aspects_set",
        feature_selection = L < 43)
x, y = load_set("test", tens = 10000)
# x_val, y_val   = load_set("val")
# x_test, y_test = load_set("test")
# print(y.shape, x.shape)
# print(np.unique(y, return_counts = True))
# exit(0)

def print_weights(model):
    for m in model.named_parameters():
        print(m)
        
def print_grad(model):
    model_sequential = model.model.model
    for m in model_sequential:
        try:
            print(m.weight.grad)
        except:
            pass
            
def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight.data)
        # torch.nn.init.constant_(layer.weight.data, 1/layer.weight.data.numel())
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)

def evaluate(model, x_set, y_set, verbose: bool = False, weights_path: str = None):
    if weights_path is not None:
        # "sapo/exp7-10K epochs all train set/exp7_weights.pt"
        model.load_state_dict(torch.load(weights_path))
    model.train(False)
    preds = []
    for i in range(len(x_set)):
        pred = model(x_set[i])
        if verbose: print(pred, y_set[i])
        preds.append( np.round([float(pred)]) )
    return {"loss": LOSS(torch.Tensor(np.array(preds))), 
            "accur": accuracy_score(y_set, preds)*100, 
            "prec": precision_score(y_set, pred)*100,
            "recall": recall_score(y_set, pred)*100}
    
def evaluate_instances(model, loader, weights_path: str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = load_set("test")
    for i in range(len(x_test)):
        pred = model.get_instance_predictions(x_test[i])
        print(model(x_test[i]), y_test[i])
        if y_test[i] != 10:
            for r in REGIONS:
                r = REGIONS[r]
                print(f"{r}\t{pred[r]:.4f}\t{int(np.round(pred[r]))}  {y_instance[i][r]}")
        input()
    exit(0)
    
def instance_level_performance(model, loader, weights_path: str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = [], []
    for x, y in loader.get_set("test"):
        x_test.append(torch.cat((x[odd],x[even]), dim = 1))
        y_test.append(y)
    y_pred, y_true, positive = {REGIONS[r]: [] for r in REGIONS}, {REGIONS[r]: [] for r in REGIONS}, {REGIONS[r]: 0 for r in REGIONS}
    for i in range(len(x_test)):
        pred = model.get_instance_predictions(x_test[i])
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
    
def print_instance_level_performance(model, loader, weights_path: str = None):
    performance, positive = instance_level_performance(model, loader, weights_path)
    print("metric", end = "\t")
    for r in REGIONS: print(f"{REGIONS[r]} ({positive[REGIONS[r]]})", end = "\t")
    for metric in performance:
        print()
        print(metric, end = "\t")
        for r in performance[metric]:
            print(f"{performance[metric][r]*100:.2f}", end = "\t")
    print()

def instance_level_f1(model, loader, weights_path: str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = [], []
    for x, y in loader.get_set("test"):
        x_test.append(torch.cat((x[odd],x[even]), dim = 1))
        y_test.append(y)
    y_pred, y_true = [], []
    for i in range(len(x_test)):
        pred = model.get_instance_predictions(x_test[i])
        for r in REGIONS:
            r = REGIONS[r]
            y_pred.append( int(np.round(pred[r])) )
            y_true.append( 0 if y_test[i]==10 else y_instance[i][r] )
    return f1_score(y_true = y_true, y_pred = y_pred)

class Model(torch.nn.Module):
    def __init__(self, bias = True, T = 64):
        super().__init__()
        self.T = T
        self.model = torch.nn.Sequential(
            torch.nn.Linear(L*2,2, bias = bias),
            # torch.nn.ReLU(inplace = True),
            # torch.nn.Linear(32,16, bias = bias),
            # torch.nn.ReLU(inplace = True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2,1, bias = bias)
        )
        # self.model = torch.nn.Linear(L*2,1, bias = bias)
    def __call__(self, x):
        x = self.model(x)
        x = x * self.T
        x = torch.sigmoid(x)
        return x
        
class ModelBag(torch.nn.Module):
    def __init__(self, share_weights: bool):
        super().__init__()
        self.share_weights = share_weights
        if self.share_weights:
            self.model = Model()
        else:
            self.model = torch.nn.ModuleList([Model() for i in range(N)])
        self.regions = [REGIONS[r] for r in REGIONS]
        assert len(self.regions) == N
    def __call__(self, instances):
        if self.share_weights:
            x = self.model(instances)
            x = N - x.sum()
        else:
            x = N
            for i in range(N):
                x = x - self.model[i](instances[i])
        return x
    def evaluate(self, instances):
        if self.share_weights:
            x = self.model(instances)
            x = torch.round(x)
            x = N - x.sum()
        else:
            x = N
            for i in range(N):
                x = x - torch.round(self.model[i](instances[i]))
        return x
    def get_instance_predictions(self, x):
        if self.share_weights:
            return {self.regions[i]: float(self.model(x[i])) for i in range(N)}
        return {self.regions[i]: float(self.model[i](x[i])) for i in range(N)}

model = ModelBag(share_weights = False)
# model.load_state_dict(torch.load("sapo/exp_2N-default settings (high F1, low accur)/exp19_weights.pt"))
# model.apply(initialize_weights)
# print_instance_level_performance(model, loader, weights_path = "exp20_weights.pt")
# print_instance_level_performance(model, loader, weights_path = "sapo/exp_2N-default settings (high F1, low accur)/exp19_weights.pt")
# exit()
# evaluate_instances(model, loader, weights_path = "exp19_weights.pt")


train_optimizer = OPTIMIZER(model.parameters(), lr = LR, weight_decay = WD) #, momentum = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(train_optimizer, step_size = STEP_SIZE, gamma = 0.1)
model.train(True)
writer = SummaryWriter("sapo")
# for epoch in tqdm(range(EPOCHS)):
for epoch in range(EPOCHS):
    total_loss = 0
    preds = []
    # for i in np.random.choice(len(x), len(x), replace = False):
    for i in range(len(x)):
        train_optimizer.zero_grad()
        pred = model(x[i])
        loss = LOSS(pred, y[i])
        loss.backward()              # compute the loss and its gradients
        train_optimizer.step()       # adjust learning weights
        total_loss += float(loss)
        preds.append( np.round(float(pred)) )
    total_loss  = total_loss / len(x)
    accuracy    = accuracy_score(y, preds)*100
    lr          = scheduler.get_last_lr()[0]
    f1_test     = instance_level_f1(model, loader)
    # val_accuracy, val_loss = evaluate(model, x_val, y_val)
    # test_accuracy, test_loss = evaluate(model, x_test, y_test)
    print(epoch, lr)
    print(f"test f1 score: {f1_test*100:.4f}")
    print("set\tloss\taccuracy")
    print(f"train\t{total_loss:.4f}\t{accuracy:.3f}")
    # print(f"val\t{val_loss:.4f}\t{val_accuracy:.3f}")
    # print(f"test\t{test_loss:.4f}\t{test_accuracy:.3f}")
    print()
    writer.add_scalar(f"LR", lr, epoch)
    writer.add_scalar(f"loss/train", total_loss, epoch)
    writer.add_scalar(f"accuracy/train", accuracy, epoch)
    writer.add_scalar(f"f1/test", f1_test, epoch)
    # writer.add_scalar(f"loss/val", val_loss, epoch)
    # writer.add_scalar(f"accuracy/val", val_accuracy, epoch)
    # writer.add_scalar(f"loss/test", test_loss, epoch)
    # writer.add_scalar(f"accuracy/test", test_accuracy, epoch)
    scheduler.step()
torch.save(model.state_dict(), "exp20_weights.pt")
