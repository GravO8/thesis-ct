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
LOSS        = torch.nn.MSELoss()
OPTIMIZER   = torch.optim.Adam
EPOCHS      = 300


def create_model():
    diff                = ABSDiff()
    model               = ASPECTSInstanceModel([43,32,16,1], return_probs = False)
    instance_classifier = ASPECTSInstanceClassifier(diff, model)
    bag_classifier      = ASPECTSBagClassifier(instance_classifier, share_weights = True)
    return bag_classifier
    
    
def get_predictions(loader, model, set_name: str):
    model.train(False)
    y_pred, y_true = [], []
    for x, y in loader.get_set(set_name):
        pred = model(x)
        y_pred.append(pred)
        y_true.append(y)
    return y_pred, y_true
    
    
def model_performance(y_pred, y_true):
    # loss, accuracy and confusion matrix
    # rows = y_true
    # cols = y_pred
    loss      = float(LOSS(y_pred, y_true))
    accuracy  = accuracy_score  (y_true = y_true, y_pred = y_pred)
    confusion = confusion_matrix(y_true = y_true, y_pred = y_pred)
    return {"loss": loss, "accuracy": accuracy, "confusion_matrix": confusion}
    

def record_performance(writer, loader, model, y_pred, y_true):
    def record_performance_aux(set, y_pred, y_true):
        performance = model_performance(y_pred, y_true)
        writer.add_scalar(f"loss/{set}", performance["loss"], epoch)
        writer.add_scalar(f"accuracy/{set}", performance["accuracy"], epoch)
        print(f"  {set} loss", performance["loss"])
        print(f"  {set} accuracy", performance["accuracy"])
        print(f"  {set} confusion matrix\n", performance["confusion_matrix"])
    record_performance_aux("train", y_pred, y_true)
    for s in ("val", "test"):
        y_pred, y_true = get_predictions(loader, model, s)
        record_performance_aux(s, y_pred, y_true)


def train(loader, model):
    model.train(True)
    writer           = SummaryWriter()
    train_optimizer  = OPTIMIZER(model.parameters(), lr = LR, weight_decay = WD)
    for epoch in range(EPOCHS):
        y_pred, y_train = [], []
        print(f"\n\n\n- Epoch {epoch}/{EPOCHS} ---------------------------------------------------------------------")
        for x, y in loader.get_set("train"):
            pred = model(x)
            loss = LOSS(pred, y)
            loss.backward()              # compute the loss and its gradients
            train_optimizer.step()       # adjust learning weights
            y_pred.append(pred)
            y_train.append(y)
        record_performance(writer, loader, model, y_pred, y_train)
        model.train(True)
        writer.flush()


if __name__ == "__main__":
    dirname = "../../../data/gravo"
    # dirname = "/media/avcstorage/gravo/"
    loader = ASPECTSMILLoader("ncct_radiomic_features.csv", "all", dirname = dirname)
    model = create_model()
    train(loader, model)
