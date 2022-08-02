import torch


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
    # print(model.instance_classifer.model[0].weight.grad)
    # print(model.instance_classifer.model[2].weight.grad)
    print(model.instance_classifer1.model[0].weight.grad)
    print(model.instance_classifer2.model[0].weight.grad)
    print(model.instance_classifer3.model[0].weight.grad)
def weights_init(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 1)

class InstanceClassifier(torch.nn.Module):
    def __init__(self, bias = True):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2,2, bias = bias),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2,1, bias = bias)
        )
    def __call__(self, x):
        x = self.model(x)
        x = torch.sign(x)
        x = torch.relu(x)
        return x
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_classifer1 = InstanceClassifier()
        self.instance_classifer2 = InstanceClassifier()
        self.instance_classifer3 = InstanceClassifier()
    def __call__(self, x):
        x1 = self.instance_classifer1(x[0])
        x2 = self.instance_classifer2(x[1])
        x3 = self.instance_classifer3(x[2])
        # x = 3 - x.sum()
        x = - x1 - x2 - x3 + 3
        return x

model = Model()
weights_init(model)

LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
LR          = 0.02 # learning rate
EPOCHS      = 1000

# for p in model.parameters():
#     print(p)
# exit()

train_optimizer  = OPTIMIZER(model.parameters(), lr = LR) #, momentum = 0.9)
model.train(True)
for epoch in range(EPOCHS):
    preds = []
    for i in range(len(x)):
        train_optimizer.zero_grad()
        pred = model(x[i].reshape(3,2))
        loss = LOSS(pred, y[i])
        loss.backward()              # compute the loss and its gradients
        train_optimizer.step()       # adjust learning weights
        # preds.append(pred)
        print(loss)
        print_grad(model)
        input()
    
    # if epoch > 0:
    #     print(f"\n\n\n- Epoch {epoch}/{EPOCHS} ---------------------------------------------------------------------")
    #     model.train(False)
    #     loss = LOSS(torch.tensor(preds).squeeze(), y.squeeze())
    #     # print(preds, float(loss))
    #     print_grad(model)
    #     input()
    #     model.train(True)
