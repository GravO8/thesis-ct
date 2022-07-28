import torch
from aspects_instance_classifier import ASPECTSInstanceClassifier
from torch.utils.tensorboard import SummaryWriter
from aspects_side_comparator import ABSDiff
from aspects_instance_model import ASPECTSInstanceModel
from aspects_bag_classifier import ASPECTSBagClassifier
from aspects_mil_loader import ASPECTSMILLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torchsummary import summary


LR          = 0.0005 # learning rate
WD          = 0.0001 # weight decay
LOSS        = torch.nn.NLLLoss()
OPTIMIZER   = torch.optim.Adam
EPOCHS      = 300


def create_model():
    diff                = ABSDiff()
    model               = ASPECTSInstanceModel([66,32,16,1], return_probs = False)
    instance_classifier = ASPECTSInstanceClassifier(diff, model)
    bag_classifier      = ASPECTSBagClassifier(instance_classifier, share_weights = True)
    return bag_classifier
    
    
def get_predictions(loader, model, set_name: str):
    model.train(False)
    set          = loader.get_set(set_name)
    x_set, y_set = set["x"], set["y"]
    out          = []
    for i in range(len(y_set)):
        x, y = x_set[i], y_set[i]
        pred = model(x)
        out.append( np.argmax(pred) )
    return out, y_set
    
    
def model_performance(y_pred, y_true):
    # loss, accuracy and confusion matrix
    # rows = y_true
    # cols = y_pred
    loss      = float(LOSS(y_pred, y_true))
    accuracy  = accuracy_score  (y_true = y_true, y_pred = y_pred)
    confusion = confusion_matrix(y_true = y_true, y_pred = y_pred)
    return {"loss": loss, "accuracy": accuracy, "confusion_matrix": confusion}
    

def record_performance(loader, model, y_true, y_pred):
    # TODO


def train(loader, model):
    model.train(True)
    writer           = SummaryWriter()
    train_optimizer  = OPTIMIZER(model.parameters(), lr = LR, weight_decay = WD)
    train_set        = loader.get_set("train")
    x_train, y_train = train_set["x"], train_set["y"]
    for epoch in range(EPOCHS):
        y_pred = []
        print(f"\n\n\n- Epoch {epoch}/{EPOCHS} ---------------------------------------------------------------------")
        for i in range(len(y_train)):
            x, y = x_train[i], y_train[i]
            pred = model(x)
            loss = LOSS(pred, y)
            loss.backward()              # compute the loss and its gradients
            train_optimizer.step()       # adjust learning weights
            y_pred.append( np.argmax(pred) )
        record_performance(loader, model, y_train, y_pred)
        model.train(True)


if __name__ == "__main__":
    # loader = ASPECTSMILLoader("ncct_radiomic_features.csv", 
    #                         "all", dirname = "../../../data/gravo")
    model = create_model()
