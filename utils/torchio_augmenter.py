import torchio, os
import pandas as pd
from tqdm import tqdm

# Augmentations where done with:
# augmentations = [torchio.RandomAffine(scales = 0, translation = 0, degrees = 10, center = "image"),
#                  torchio.RandomElasticDeformation(),
#                  torchio.RandomNoise(mean = 5, std = 2),
#                  torchio.RandomFlip("lr")]


class Augmenter:
    def __init__(self, dataset_filename: str, augmentation: str, data_dir: str = ""):
        '''
        TODO
        '''
        self.dataset      = pd.read_csv(os.path.join(data_dir, dataset_filename))
        self.data_dir     = data_dir
        self.augmentation = augmentation
        
    def augment(self, ct_type: str):
        '''
        TODO
        '''
        self.ct_dir = os.path.join(self.data_dir, ct_type)
        for _, row in tqdm(self.dataset.iterrows()):
            if row["set"] == "train":
                patient_id          = row["patient_id"]
                augmentation_name   = self.augmentation.__class__.__name__
                filename            = os.path.join(self.ct_dir, f"{patient_id}-{augmentation_name}.nii")
                if not os.path.isfile(filename):
                    subject = self.create_subject(patient_id)
                    if augmentation_name == "RandomFlip":
                        augmented = self.augment_flip(subject)
                    else:
                        augmented = self.augmentation(subject)
                    augmented["ct"].save(filename)
                    
    def augment_flip(self, subject):
        '''
        TODO
        Because the augmentations are random (after all, the augmentation is 
        called RandomFlip), sometimes the "flipped" scan is actualy in the same
        position it was originally. If we call the augmentation enough times, 
        the flipped scan will eventually show up
        '''
        augmented = self.augmentation(subject)
        while (augmented["ct"][torchio.DATA].numpy() == subject["ct"][torchio.DATA].numpy()).all():
            augmented = self.augmentation(subject)
        return augmented
            
    def create_subject(self, patient_id: str):
        '''
        TODO
        '''
        path    = os.path.join(self.ct_dir, f"{patient_id}.nii")
        subject = torchio.Subject(ct  = torchio.ScalarImage(path))
        return subject


if __name__ == "__main__":
    augmenter = Augmenter("dataset_cta.csv", 
                        torchio.RandomFlip("lr"),
                        data_dir = "/media/avcstorage/gravo")
    augmenter.augment("CTA")
    
