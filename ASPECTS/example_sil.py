import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from aspects_mil_loader import ASPECTSMILLoader, REGIONS

N           = 10  # number of classes
L           = 43  # instance size
LOSS        = torch.nn.BCELoss()
OPTIMIZER   = torch.optim.Adam
LR          = 0.001 # learning rate
WD          = 0.001
EPOCHS      = 200
STEP_SIZE   = 80 # step size to update the LR


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
        y.append(gt)
        x.append( torch.abs(x_sample[odd]-x_sample[even]).numpy() )
    x = torch.Tensor(np.array(x))
    y = torch.Tensor(y).view(-1,1)
    return x, y
    
dirname = "../../../data/gravo"
# dirname = "/media/avcstorage/gravo/"
loader = ASPECTSMILLoader("ncct_radiomic_features.csv", "all", 
        normalize = True, dirname = dirname, set_col = "instance_aspects_set",
        feature_selection = False)
x, y = load_set("test", tens = 1000)
print(y.shape, x.shape)
    
class Model(torch.nn.Module):
    def __init__(self, bias = True, T = 64):
        super().__init__()
        self.T = T
        self.model = torch.nn.Sequential(
            torch.nn.Linear(L,2, bias = bias),
            torch.nn.ReLU(inplace = True),
            # torch.nn.Sigmoid(),
            torch.nn.Linear(2,1, bias = bias)
        )
        # self.model = torch.nn.Linear(L,1, bias = bias)
    def __call__(self, x):
        x = self.model(x)
        x = x * self.T
        x = torch.sigmoid(x)
        return x
        
class ModelBag(torch.nn.Module):
    def __init__(self, instance_models):
        super().__init__()
        self.share_weights = False
        self.model = torch.nn.ModuleList(instance_models)
        self.regions = REGIONS
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

def evaluate(model, x_set, y_set, verbose: bool = False, weights_path: str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    model.train(False)
    preds, y_trues = [], []
    for i in range(len(x_set)):
        pred = model(x_set[i])
        if verbose: print(pred, y_set[i])
        preds.append( np.round(float(pred)) )
        y_trues.append( y_set[i].numpy()[0] )
    return {"accur": accuracy_score(y_trues, preds)*100, 
            "confusion_matrix": confusion_matrix(y_trues, preds)}


REGIONS = [REGIONS[r] for r in REGIONS]
y_instance = loader.get_test_instance_labels()
models = []
for r in range(len(REGIONS)):
    model           = Model()
    train_optimizer = OPTIMIZER(model.parameters(), lr = LR, weight_decay = WD)
    scheduler       = torch.optim.lr_scheduler.StepLR(train_optimizer, step_size = STEP_SIZE, gamma = 0.1)
    model.train(True)
    for epoch in range(EPOCHS):
        total_loss = 0
        preds, y_trues = [], []
        for i in range(len(x)):
            train_optimizer.zero_grad()
            pred   = model(x[i,r])
            y_true = 0 if y[i]==10 else y_instance[i][REGIONS[r]]
            loss   = LOSS(pred, torch.Tensor([y_true]))
            loss.backward()              # compute the loss and its gradients
            train_optimizer.step()       # adjust learning weights
            total_loss += float(loss)
            preds.append( np.round(float(pred)) )
            y_trues.append( y_true )
        total_loss  = total_loss / len(x)
        accuracy    = accuracy_score(y_trues, preds)*100
        precision   = precision_score(y_trues, preds)*100
        recall      = recall_score(y_trues, preds)*100
        lr          = scheduler.get_last_lr()[0]
    #     if epoch % 10 == 0:
    #         print(epoch, lr)
    #         print("region\tloss\taccur\tprecis\trecall")
    #         print(f"{REGIONS[r]}\t{total_loss:.4f}\t{accuracy:.3f}\t{precision:.3f}\t{recall:.3f}")
    #         print()
    # input()
    # print("-----------------------------------------------------------------------------------------")
    models.append(model)
    
model = ModelBag(models)
x_train, y_train = load_set("train", tens = 1000)
performance = evaluate(model, x_train, y_train)
for metric in performance:
    print(metric)
    print(performance[metric])
