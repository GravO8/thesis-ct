import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

i_positive  = 0
DEBUG       = False
N           = 10  # number of classes
L           = 43   # instance size
if DEBUG:
    y = [7,10,9,10,6,8,10,10,8,10,10,9,8,9,10,10,7,10,9,10,10,7,9,10,9,10,9,10,8,10,8,10,10,10,10,10,7,10,10,10,10,10,10,8,8,10,10,10,10,9,10,10,8,10,10,10,10,7,9,9,10,10,8,10,10,10,10,9,10,7,10,10,9,10,10,10,6,10,10,10,10,10,6,10,9,10,9,10,10,10,10,10,10,6,10,6,10,6,4,9,10,9,7,10,10,9,10,10,10,10,10,10,10,10,8,10,10,10,8,10,10,10,10,10,10,10,7,10,10,10,10,10,8,10,7,7,10,7,10,10,8,7,8,6,10,7,10,10,9,10,10,10,10,10,10,10,10,9,7,9,10,10,10,10,7,8,9,10,10,6,10,9,6,9,10,10,10,10,10,8,8,10,8,10,10,10,7,10,7,6,10,7,10,7,10,10,9,10,10,10,5,10,9,10,10,9,10,10,9,10,10,6,7,7,10,6,10,10,8,10,10,8,7,10,8,10,10,8,10,10,10,9,10,10,10,10,10,10,10,10,10,10,9,10,10,10,10,10,10,2,10,10,10,10,8,10,7,10,10,10,8,10,10,8,10,10,10,10,10,10,10,10,10,10,10,10,8,10,10,8]
    # N_samples = 15
    # x, y = [], []
    # for i in range(N_samples):
    #     gt = np.random.randint(0,N+1)
    #     y.append(gt)
    #     positive = np.random.choice(N, gt, replace = False)
    #     x_sample = []
    #     for i in range(N):
    #         if i in positive:
    #             x_sample.append([0]*L)
    #         else:
    #             x_sample.append([0]*L)
    #             x_sample[-1][0] = 1
    #     x.append(x_sample)
    x = []
    for i in range(len(y)):
        positive = np.random.choice(N, y[i], replace = False)
        x_sample = []
        for i in range(N):
            if i in positive:
                x_sample.append([0]*L)
            else:
                x_sample.append([0]*L)
                x_sample[-1][i_positive] = 1
        x.append(x_sample)
else:
    from aspects_mil_loader import ASPECTSMILLoader
    dirname = "../../../data/gravo"
    loader = ASPECTSMILLoader("ncct_radiomic_features.csv", "all", 
            normalize = True, dirname = dirname)
    x, y = [], []
    odd, even = range(0,N*2,2), range(1,N*2,2)
    for x_sample, gt in loader.get_set("train"):
        if gt != 10:
            y.append(gt)
            x.append( torch.abs(x_sample[odd]-x_sample[even]).numpy() )
x = torch.Tensor(np.array(x))
y = torch.Tensor(y).view(-1,1)
print(y.shape, x.shape)
# print(y)
# exit(0)
# for i in range(len(x)):
#     print(x[i], y[i])
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
    # print(model.model.model[0].weight.grad)
    # print(model.model.model[2].weight.grad)

class Model(torch.nn.Module):
    def __init__(self, simple = True, bias = True, T = 6):
        super().__init__()
        self.T = T
        if simple:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(L,2, bias = bias),
                torch.nn.Sigmoid(),
                torch.nn.Linear(2,1, bias = bias)
            )
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(L,32, bias = bias),
                torch.nn.ReLU(inplace = True),
                torch.nn.Linear(32,16, bias = bias),
                torch.nn.ReLU(inplace = True),
                torch.nn.Linear(16,8, bias = bias),
                torch.nn.ReLU(inplace = True),
                torch.nn.Linear(8,1, bias = bias)
            )
    def __call__(self, x):
        x = self.model(x)
        x = x * self.T
        x = torch.sigmoid(x)
        return x
        
class ModelBag(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
    def __call__(self, x):
        x = self.model(x)
        x = N - x.sum()
        return x

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight.data)
        # torch.nn.init.constant_(layer.weight.data, 1/layer.weight.data.numel())
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
model = ModelBag()
# torch.save(model.state_dict(), "good_weights.pt")
model.load_state_dict(torch.load("good_weights.pt"))
# model.apply(initialize_weights)

LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
LR          = 0.1 # learning rate
EPOCHS      = 1000

train_optimizer  = OPTIMIZER(model.parameters(), lr = LR) #, momentum = 0.01)
model.train(True)
writer = SummaryWriter("sapo")
for epoch in range(EPOCHS):
    total_loss = 0
    # for i in np.random.choice(len(x), len(x), replace = False):
    preds = []
    for i in range(len(x)):
        train_optimizer.zero_grad()
        pred = model(x[i])
        loss = LOSS(pred, y[i][0])
        loss.backward()              # compute the loss and its gradients
        train_optimizer.step()       # adjust learning weights
        # print_grad(model)
        # print_weights(model)
        # print(pred, y[i][0])
        # print("\n\n")
        # input()
        total_loss += float(loss)
        preds.append( np.round(float(pred)) )
        # if epoch > 10:
        #     model.train(False)
        #     a = [0]*L
        #     a[i_positive] = 1
        #     # pred = model.model(torch.Tensor([[0]*L, a]))
        #     # loss = LOSS(pred, torch.Tensor([0, 1]).view(-1,1))
        #     # print(pred, float(loss))
        #     input()
        #     model.train(True)
        # if (not DEBUG) and (epoch > 0):
            # print(pred, y[i][0])
            # print_weights(model)
            # print_grad(model)
            # input()
    total_loss = total_loss / len(x)
    accuracy = accuracy_score(y, preds)
    writer.add_scalar(f"loss", total_loss, epoch)
    writer.add_scalar(f"accuracy", accuracy, epoch)
    print(epoch, total_loss, accuracy)
