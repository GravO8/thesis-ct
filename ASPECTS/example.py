import torch

x = torch.Tensor([[0,0], [0,1], [1,0], [1,1]])
y = torch.Tensor([1, 0, 0, 1]).view(-1,1)

def print_weights(model):
    for m in model.named_parameters():
        print(m)
        
def print_grad(model):
    print(model.model[0].weight.grad)
    print(model.model[2].weight.grad)

class Model(torch.nn.Module):
    def __init__(self, bias = True):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2,2, bias = bias),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2,1, bias = bias)
        )
    def __call__(self, x):
        return self.model(x)

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
model = Model()
model.apply(initialize_weights)

LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.SGD
LR          = 0.2 # learning rate
EPOCHS      = 100

train_optimizer  = OPTIMIZER(model.parameters(), lr = LR, momentum = 0.9)
model.train(True)
for epoch in range(EPOCHS):
    # train_optimizer.zero_grad()
    # pred = model(x)
    # loss = LOSS(pred, y)
    # loss.backward()              # compute the loss and its gradients
    # train_optimizer.step()       # adjust learning weights
    for i in range(4):
        train_optimizer.zero_grad()
        pred = model(x[i])
        loss = LOSS(pred, y[i])
        loss.backward()              # compute the loss and its gradients
        train_optimizer.step()       # adjust learning weights
        print(x[i], y[i], pred)
        print(loss)
        print_grad(model)
        # input()
    
    if epoch > 0:
        print(f"\n\n\n- Epoch {epoch+1}/{EPOCHS} ---------------------------------------------------------------------")
        model.train(False)
        pred = model(x)
        loss = LOSS(pred, y)
        print(pred, float(loss))
        print_grad(model)
        # input()
        model.train(True)
