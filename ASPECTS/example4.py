import torch, numpy as np


x = torch.Tensor([  [[0,0],[1,1],[1,1]], 
                    [[0,0],[1,1],[1,0]], 
                    [[1,0],[1,0],[0,1]], 
                    [[0,1],[0,0],[1,0]], 
                    [[1,1],[1,0],[1,0]], 
                    [[0,0],[0,0],[0,1]], 
                    [[0,1],[1,1],[0,1]]])
y = torch.Tensor([3,2,0,1,1,2,1]).view(-1,1)

def print_weights(model):
    for m in model.named_parameters():
        print(m)
        
def print_grad(model):
    print(model.model.model[0].weight.grad)
    print(model.model.model[2].weight.grad)

class Model(torch.nn.Module):
    def __init__(self, bias = True, T = 6):
        super().__init__()
        self.T = T
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2,2, bias = bias),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2,1, bias = bias)
        )
    def __call__(self, x):
        x = self.model(x)
        x = x * self.T
        x = torch.nn.functional.sigmoid(x)
        # x = torch.sign(x)
        # x = torch.relu(x)
        return x
        
class ModelBag(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
    def __call__(self, x):
        x = self.model(x)
        x = 3 - x.sum()
        return x

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
model = ModelBag()
model.apply(initialize_weights)

LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
LR          = 0.2 # learning rate
EPOCHS      = 100

train_optimizer  = OPTIMIZER(model.parameters(), lr = LR) #, momentum = 0.01)
model.train(True)
for epoch in range(EPOCHS):
    # train_optimizer.zero_grad()
    # pred = model(x)
    # loss = LOSS(pred, y)
    # loss.backward()              # compute the loss and its gradients
    # train_optimizer.step()       # adjust learning weights
    total_loss = 0
    for i in np.random.choice(len(x), len(x), replace = False):
        train_optimizer.zero_grad()
        pred = model(x[i])
        # print(x[i], y[i], pred)
        loss = LOSS(pred, y[i])
        loss.backward()              # compute the loss and its gradients
        train_optimizer.step()       # adjust learning weights
        # print(loss)
        # print_grad(model)
        total_loss += float(loss)
        if epoch > 40:
            model.train(False)
            pred = model.model(torch.Tensor([[0,0], [0,1], [1,0], [1,1]]))
            loss = LOSS(pred, torch.Tensor([0, 1, 1, 0]).view(-1,1))
            print(pred, float(loss))
            # print_grad(model)
            input()
            model.train(True)
    print(total_loss / len(x))
