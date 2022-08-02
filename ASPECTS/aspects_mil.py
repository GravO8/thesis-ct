import torch
from aspects_instance_classifier import ASPECTSInstanceClassifier
from torch.utils.tensorboard import SummaryWriter
from aspects_side_comparator import ABSDiff
from aspects_instance_model import ASPECTSInstanceModel
from aspects_bag_classifier import ASPECTSBagClassifier
from aspects_mil_loader import ASPECTSMILLoader, ASPECTSMILLoaderBebug
from sklearn.metrics import accuracy_score, confusion_matrix
from torchsummary import summary


LR          = .01 # learning rate
WD          = 0.0001 # weight decay
LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
EPOCHS      = 2000


def create_model():
    diff                = ABSDiff()
    model               = ASPECTSInstanceModel([1,2,2,1], return_probs = False)
    instance_classifier = ASPECTSInstanceClassifier(diff, model)
    bag_classifier      = ASPECTSBagClassifier(instance_classifier, share_weights = True)
    bag_classifier.apply(initialize_weights)
    return bag_classifier
    
    
def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
    
    
def get_predictions(loader, model, set_name: str):
    model.train(False)
    y_pred, y_true = [], []
    for x, y in loader.get_set(set_name):
        pred = model(x)
        y_pred.append(float(pred))
        y_true.append(float(y))
    return y_pred, y_true
    
    
def model_performance(y_pred, y_true):
    # loss, accuracy and confusion matrix
    # rows = y_true
    # cols = y_pred
    loss      = float(LOSS(torch.tensor(y_pred), torch.tensor(y_true)))
    accuracy  = accuracy_score  (y_true = y_true, y_pred = y_pred)
    confusion = confusion_matrix(y_true = y_true, y_pred = y_pred)
    return {"loss": loss, "accuracy": accuracy, "confusion_matrix": confusion}
    

def record_performance(epoch, writer, loader, model, y_pred, y_true):
    def record_performance_aux(set, y_pred, y_true):
        performance = model_performance(y_pred, y_true)
        writer.add_scalar(f"loss/{set}", performance["loss"], epoch)
        writer.add_scalar(f"accuracy/{set}", performance["accuracy"], epoch)
        print(f"  {set} loss", performance["loss"])
        print(f"  {set} accuracy", performance["accuracy"])
        print(f"  {set} confusion matrix\n", performance["confusion_matrix"])
    record_performance_aux("train", y_pred, y_true)
    # for s in ("val", "test"):
    #     y_pred, y_true = get_predictions(loader, model, s)
    #     record_performance_aux(s, y_pred, y_true)
    

def print_weights(model):
    for m in model.named_parameters():
        print(m)

def print_grad(model):
    print(model.instance_classifer.model.mlp[0].weight.grad)
    print(model.instance_classifer.model.mlp[2].weight.grad)
    print(model.instance_classifer.model.mlp[4].weight.grad)


def train_old(loader, model):
    model.train(True)
    writer           = SummaryWriter()
    train_optimizer  = OPTIMIZER(model.parameters(), lr = LR)
    for epoch in range(EPOCHS):
        y_pred, y_train = [], []
        loss = 0
        # print_weights(model)
        print(f"\n\n\n- Epoch {epoch+1}/{EPOCHS} ---------------------------------------------------------------------")
        for x, y in loader.get_set("train"):
            # train_optimizer.zero_grad()
            pred = model(x)
            loss += LOSS(pred, y)
            # print(loss)
            # train_optimizer.step()       # adjust learning weights
            y_pred.append(float(pred))
            y_train.append(float(y))
        loss /= loader.len("train")
        loss.backward()              # compute the loss and its gradients
        print(loss)
        print_weights(model)
        train_optimizer.step()       # adjust learning weights
        record_performance(epoch, writer, loader, model, y_pred, y_train)
        model.train(True)
        writer.flush()
        
        
def train(loader, model):
    train_optimizer  = torch.optim.SGD(model.parameters(), lr = LR)
    for epoch in range(EPOCHS):
        y_pred, y_train = [], []
        model.train(True)
        for x, y in loader.get_set("train"):
            train_optimizer.zero_grad()
            pred = model(x)
            loss = LOSS(pred, y)
            loss.backward()              # compute the loss and its gradients
            train_optimizer.step()       # adjust learning weights
            y_pred.append(float(pred))
            y_train.append(float(y))
    # if epoch > 50:
        model.train(False)
        print(f"\n\n\n- Epoch {epoch+1}/{EPOCHS} ---------------------------------------------------------------------")    
        print(LOSS(torch.tensor(y_pred), torch.tensor(y_train)))
        print_grad(model)
        # print_weights(model)
        input()


if __name__ == "__main__":
    dirname = "../../../data/gravo"
    # dirname = "/media/avcstorage/gravo/"
    loader = ASPECTSMILLoaderBebug("ncct_radiomic_features.csv", "all", 
            normalize = False, dirname = dirname)
    model = create_model()
    train(loader, model)
