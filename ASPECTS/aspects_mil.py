from aspects_instance_classifier import ASPECTSInstanceClassifier
from aspects_side_comparator import ABSDiff
from aspects_instance_model import ASPECTSInstanceModel
from aspects_bag_classifier import ASPECTSBagClassifier
from aspects_mil_loader import ASPECTSMILLoader

def create_model():
    diff                = ABSDiff()
    model               = ASPECTSInstanceModel([66,32,16,1], return_probs = False)
    instance_classifier = ASPECTSInstanceClassifier(diff, model)
    bag_classifier      = ASPECTSBagClassifier(instance_classifier, share_weights = False)
    return bag_classifier


# loader = ASPECTSMILLoader("ncct_radiomic_features.csv", 
#                         "all", dirname = "../../../data/gravo")
model = create_model()
