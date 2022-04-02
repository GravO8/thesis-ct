import torchio, os
import pandas as pd
from tqdm import tqdm


class Augmenter:
    def __init__(self, table_data_file: str, augmentation, data_dir: str = None):
        '''
        TODO
        '''
        self.table_data_file     = table_data_file
        self.data_dir            = data_dir
        self.augmentation        = augmentation
        if self.data_dir is not None:
            self.table_data_file = os.path.join(self.data_dir, self.table_data_file)
        self.table_data          = pd.read_csv(self.table_data_file)
        
    def augment(self, ct_type: str):
        '''
        TODO
        '''
        self.ct_dir = ct_type if self.data_dir is None else os.path.join(self.data_dir, ct_type)
        for _, row in tqdm(self.table_data.iterrows()):
            if row[ct_type]:
                patient_id  = row["idProcessoLocal-1"]
                subject     = self.create_subject(patient_id)
                augmented   = self.augmentation(subject)
                filename    = os.path.join(self.ct_dir, 
                    f"{patient_id}-{self.augmentation.__class__.__name__}.nii")
                augmented["ct"].save(filename)
            
    def create_subject(self, patient_id: str):
        '''
        TODO
        '''
        path    = os.path.join(self.ct_dir, f"{patient_id}.nii")
        subject = torchio.Subject(ct  = torchio.ScalarImage(path))
        return subject


if __name__ == "__main__":
    augmenter = Augmenter("table_data.csv", 
                        torchio.RandomElasticDeformation(), 
                        data_dir = "../../../data/gravo")
    augmenter.augment("NCCT")
    
