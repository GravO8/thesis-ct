import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from aspects_mil_loader import ASPECTSMILLoader, REGIONS
from tqdm import tqdm

N           = 10  # number of classes
L           = 43  # instance size
N_features  = 16
LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
LR          = 0.001 # learning rate
WD          = 0.001
EPOCHS      = 500
STEP_SIZE   = 150 # step size to update the LR
EVEN, ODD   = range(0,N*2,2), range(1,N*2,2)

dirname = "../../../data/gravo"
# dirname = "/media/avcstorage/gravo/"
loader = ASPECTSMILLoader("ncct_radiomic_features.csv", "all", 
        normalize = True, dirname = dirname)


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
    return accuracy_score(y_set, preds)*100, LOSS(torch.Tensor(np.array(preds)), y_set).numpy()
    
def evaluate_instances(model, loader, weights_path: str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = [], []
    for x, y in loader.get_set("test"):
        x_test.append(x)
        y_test.append(y)
    for i in range(len(x_test)):
        pred = model.get_instance_predictions(x_test[i])
        print(model(x_test[i]), y_test[i])
        if y_test[i] != 10:
            for r in REGIONS:
                r = REGIONS[r]
                print(f"{r}\t{pred[r]:.4f}\t{int(np.round(pred[r]))}  {y_instance[i][r]}")
                # print(r, pred[r], torch.round(pred[r]), y_instance[i][r])
        input()
    exit(0)
    
def instance_level_performance(model, loader, weights_path: str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    y_instance = loader.get_test_instance_labels()
    x_test, y_test = [], []
    for x, y in loader.get_set("test"):
        x_test.append(x)
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
    
def print_instance_level_performance(performance: dict, positive: dict):
    print("metric", end = "\t")
    for r in REGIONS: print(f"{REGIONS[r]} ({positive[REGIONS[r]]})", end = "\t")
    for metric in performance:
        print()
        print(metric, end = "\t")
        for r in performance[metric]:
            print(f"{performance[metric][r]*100:.2f}", end = "\t")
    print()

class Model(torch.nn.Module):
    def __init__(self, bias = True, T = 64):
        super().__init__()
        self.T = T
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(L,N_features, bias = bias),
            torch.nn.Sigmoid()
        )
        self.classifier = torch.nn.Linear(N_features, 1, bias = bias)
    def __call__(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.abs(x1 - x2)
        x = self.classifier(x)
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
        x1, x2 = instances[EVEN], instances[ODD]
        if self.share_weights:
            x = self.model(x1, x2)
            x = N - x.sum()
        else:
            x = N
            for i in range(N):
                x = x - self.model[i](x1[i], x2[i])
        return x
    def evaluate(self, instances):
        x1, x2 = instances[EVEN], instances[ODD]
        if self.share_weights:
            x = self.model(x1, x2)
            x = torch.round(x)
            x = N - x.sum()
        else:
            x = N
            for i in range(N):
                x = x - torch.round(self.model[i](x1[i], x2[i]))
        return x
    def get_instance_predictions(self, instances):
        x1, x2 = instances[EVEN], instances[ODD]
        if self.share_weights:
            return {self.regions[i]: float(self.model(x1[i], x2[i])) for i in range(N)}
        return {self.regions[i]: float(self.model[i](x1[i], x2[i])) for i in range(N)}

model = ModelBag(share_weights = True)
# model.apply(initialize_weights)
# evaluate_instances(model, loader, weights_path = "exp_siamese1_weights.pt")


train_optimizer = OPTIMIZER(model.parameters(), lr = LR, weight_decay = WD) #, momentum = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(train_optimizer, step_size = STEP_SIZE, gamma = 0.1)
model.train(True)
writer = SummaryWriter("sapo")
# for epoch in tqdm(range(EPOCHS)):
for epoch in range(EPOCHS):
    total_loss = 0
    preds = []
    y_true = []
    for x, y in loader.get_set("train"):
        train_optimizer.zero_grad()
        pred = model(x)
        loss = LOSS(pred, y)
        loss.backward()              # compute the loss and its gradients
        train_optimizer.step()       # adjust learning weights
        total_loss += float(loss)
        preds.append( np.round(float(pred)) )
        y_true.append( y )
    total_loss  = total_loss / len(x)
    accuracy    = accuracy_score(y_true, preds)*100
    lr          = scheduler.get_last_lr()[0]
    # val_accuracy, val_loss = evaluate(model, x_val, y_val)
    # test_accuracy, test_loss = evaluate(model, x_test, y_test)
    print(epoch, lr)
    print("set\tloss\taccuracy")
    print(f"train\t{total_loss:.4f}\t{accuracy:.3f}")
    print_instance_level_performance(*instance_level_performance(model, loader))
    # print(f"val\t{val_loss:.4f}\t{val_accuracy:.3f}")
    # print(f"test\t{test_loss:.4f}\t{test_accuracy:.3f}")
    print()
    writer.add_scalar(f"LR", lr, epoch)
    writer.add_scalar(f"loss/train", total_loss, epoch)
    writer.add_scalar(f"accuracy/train", accuracy, epoch)
    # writer.add_scalar(f"loss/val", val_loss, epoch)
    # writer.add_scalar(f"accuracy/val", val_accuracy, epoch)
    # writer.add_scalar(f"loss/test", test_loss, epoch)
    # writer.add_scalar(f"accuracy/test", test_accuracy, epoch)
    scheduler.step()
torch.save(model.state_dict(), "exp_siamese1_weights.pt")
