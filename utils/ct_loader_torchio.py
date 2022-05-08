import torch, torchio, os
import pandas as pd
import numpy as np

rescale = torchio.RescaleIntensity(out_min_max = (0, 1))

class TargetTransform:
    def __init__(self, name, fn):
        self.name   = name
        self.fn     = fn
    def __call__(self, x: int) -> int:
        return self.fn(x)
    def __repr__(self):
        return self.name


def to_subject_datasets(train, validation, test):
    '''
    TODO: update signature
    Input:  train, list of torhcio subjects objects
            validation, list of torhcio subjects objects
            test, list of torhcio subjects objects
    Output: 3 torchio SubjectsDataset created from the 3 inputed lists. Their 
            scans are rescaled so every voxel is between 0 and 1
    '''
    return  (torchio.SubjectsDataset(train, transform = rescale),
            torchio.SubjectsDataset(validation, transform = rescale),
            torchio.SubjectsDataset(test, transform = rescale))


def list_scans(path: str = None):
    '''
    Input:  path, a string with a path to a directory
    Output: list of file names inside the path given as argument, whose file extension
            is ".nii" (i.e. the NIfTI format)
    '''
    return [c for c in os.listdir(path) if c.endswith(".nii")]


def binary_mrs(label: int):
    '''
    Behaviour:  Applies the binarization function proposed in "Using Machine Learning to 
                Improve the Prediction of Functional Outcome in Ischemic Stroke Patients", 
                for the target variable (mRS).
    Input:      label, a integer
    Output:     integer (0 or 1)
    '''
    return 0 if int(label) <= 2 else 1
    
    
def to_bins(train: list):
    '''
    Input:  train, list of torchio subjects
    Output: a dictionary whose keys are the different classes and its values are 
            the list of torchio subjects that belong to that class
    '''
    bins = {}
    for t in train:
        if t["target"] in bins:
            bins[t["target"]].append(t)
        else:
            bins[t["target"]] = [t]
    return bins


def to_bin_count(l: list):
    '''
    Input:  l, a list
    Output: a dictionary whose keys are the different unique values on the list
            and their values is their respective count
    '''
    bins = {}
    for t in l:
        if t in bins:
            bins[t] += 1
        else:
            bins[t] = 1
    return bins
    

def data_distribution(train, validation, test):
    '''
    TODO update
    Input:  train, list of torhcio subjects objects
            validation, list of torhcio subjects objects
            test, list of torhcio subjects objects
    Output: a dictionary whose keys are the strings 'train', 'validation' and 'test'
            and values are the respective data data distributions of their lists
    '''
    distr               = {}
    train_target        = [t["target"] for t in train]
    distr["train"]      = to_bin_count(train_target)
    distr["validation"] = to_bin_count([t["target"] for t in validation])
    distr["test"]       = to_bin_count([t["target"] for t in test])
    for label in np.unique(train_target):
        distr[f"train_augment-{label}"] = to_bin_count([t["transform"] for t in train if t["target"] == label])
    return distr
    

class CTLoader:
    def __init__(self, table_data_file: str, ct_type: str, 
        target_transform: TargetTransform = None, has_both_scan_types: bool = False, 
        balance_test_set: bool = True, random_seed: int = None, 
        balance_train_set: bool = False, data_dir: str = None, validation_size: float = 0.2, 
        target: str = "rankin",
        transforms: list = ["RandomFlip", "RandomElasticDeformation", "RandomNoise", "RandomAffine"]):
        '''
        TODO: signature needs update
        Input:  table_data_file, string with a path to a csv file
                ct_type, a string ct_type, a string, either "NCCT" or "CTA"
                binary_label, boolean. If true, the target variable is turned into a binary variable
                has_both_scan_types, boolean. If true, only those patients that have
                both a NCCT and a CTA exam are loaded
                balance_test_set, boolean. If true, the examples in the test
                set are balanced, i.e. the k_fold_balance_test function is used
                random_seed, integer with to set the numpy random seed
                balance_train_set, boolean. If true, the trainset is balanced, i.e.
                the balance_augment_train_classes function is used
                data_dir, string with the path location of the ct_type directory and table_data_file
                validation_size, float specifying the relative size of the validation set (this percentage
                is relative to the size of the trai set, not the whole dataset)
        '''
        assert ct_type in ("NCCT", "CTA"), f"CT_loader.__init__: ct_type must be NCCT or CTA"
        assert 0 < validation_size < 1, "CT_loader.__init__: validation_size must be between 0 and 1"
        self.table_data_file        = table_data_file
        self.ct_type                = ct_type
        self.target_transform       = target_transform
        self.has_both_scan_types    = has_both_scan_types
        self.balance_test_set       = balance_test_set
        self.random_seed            = random_seed
        self.balance_train_set      = balance_train_set
        self.data_dir               = data_dir
        self.validation_size        = validation_size
        self.target                 = target
        self.data_distr             = {}
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.transforms = transforms
        self.init_table_data()
        
    def create_subject(self, row, transform: str = None):
        '''
        TODO: signature needs update
        Input:  row, pandas DataFrame row of the patient to be loaded into a subject torchio object
        Output: torchio object with the loaded scan
        '''
        if self.data_dir is not None:
            ct_type = os.path.join(self.data_dir, self.ct_type)
        target      = int(row[self.target])
        patient_id  = int(row["idProcessoLocal"])
        if transform is None:
            path = os.path.join(ct_type, f"{patient_id}.nii")
        else:
            path = os.path.join(ct_type, f"{patient_id}-{transform}.nii")
        subject = torchio.Subject(
                ct          = torchio.ScalarImage(path),
                patient_id  = patient_id,
                target      = target,
                transform   = "original" if transform is None else transform)
        return subject
        
    def augment(self, to_augment, n_augment):
        '''
        TODO
        to_augment - list to augment
        n_augment  - size of the list after the augmentation 
        '''
        bin_size    = len(to_augment)
        to_add      = n_augment - bin_size
        if to_add > 0:
            transforms_to_add = [to_add // len(self.transforms)] * len(self.transforms)
            ids = [str(t["patient_id"]) for t in to_augment]
            for i in range(to_add % len(self.transforms)):
                transforms_to_add[i] += 1
            for i in range(len(transforms_to_add)):
                for id in np.random.choice(ids, size = transforms_to_add[i], replace = False):
                    row = self.table_data[self.table_data["idProcessoLocal"] == id]
                    to_augment.append( self.create_subject(row, self.transforms[i]) )
        else:
            # If the classes are too unbalanced, then maybe some classes don't 
            # need any augmentations and should actually be undersampled
            to_augment = np.random.choice(to_augment, size = n_augment, replace = False)
        return to_augment
        
    def init_table_data(self):
        '''
        Output: pandas DataFrame with the table_data_file csv loaded that satisfies the
                inputted settings
        '''
        if self.data_dir is not None:
            self.table_data_file = os.path.join(self.data_dir, self.table_data_file)
        table_data = pd.read_csv(self.table_data_file)
        table_data = table_data[table_data[self.target] != "missing"]
        table_data = table_data[table_data[self.target] != "None"]
        table_data = table_data[table_data[self.ct_type] != "missing"]
        if self.has_both_scan_types:
            table_data = table_data[(table_data["NCCT"] != "missing") & (table_data["CTA"] != "missing")]
        if self.target_transform is not None:
            table_data[self.target] = [self.target_transform(t) for t in table_data[self.target].values]
        random_permutation  = np.random.permutation( len(table_data) )
        table_data          = table_data.iloc[random_permutation]
        self.table_data     = table_data
        
    def to_dict(self):
        params  = ["ct_type", "has_both_scan_types", "balance_test_set", "random_seed", 
                "balance_train_set", "validation_size", "target", "target_transform",
                "transforms"]
        dict    = {param: str(self.__dict__[param]) for param in params}
        dict["data_distribution"] = self.data_distr
        return dict
        
    def subject_dataset(self, train_size: float = 0.75):
        '''
        Input:  train_size, float specifying the relative size of the train set (the remaining
                percentage is reserved for testing)
        Output: 3 torchio SubjectsDatasets, with train, validation and test patients, 
                respectivly that satisfy the inputted settings
        '''
        assert 0 < train_size < 1, "subject_dataset: train_size must be between 0 and 1"
        subjects    = [self.create_subject(row) for _, row in self.table_data.iterrows()]
        train, test = self.split_set(subjects, 
                                    split_size = train_size, 
                                    balance_first_set = False,
                                    balance_second_set = self.balance_test_set)
        train, validation = self.split_set(train,
                                        split_size = 1 - self.validation_size,
                                        balance_first_set = self.balance_train_set,
                                        balance_second_set = True)
        self.data_distr = data_distribution(train, validation, test)
        return to_subject_datasets(train, validation, test)
        
    def k_fold(self, k: int = 5):
        '''
        Input:  k, integer with the number of folds
        Output: generator that yields k pairs of torchio SubjectsDatasets, with train and
                test, respectivly, that match the inputted settings
        Note:   Classes that aren't the minority class will be undersampled to generate
                a balanced test set. To make sure every example is actually tested, the
                user can run k_fold with different initial random_seeds
        '''
        assert k > 1, "k_fold: for k=1 use create_subject_dataset with train_size = 0.5 instead"
        if self.balance_test_set:
            class_bins  = np.bincount( self.table_data[self.target] )
            min_class   = np.min(class_bins)
            n_test      = min_class // k
            subjects    = {label:[] for label in range(len(class_bins))}
            for _, row in self.table_data.iterrows():
                subjects[int(row[self.target])].append( self.create_subject(row) )
            for i in range(k):
                test, train = [], []
                for label in range(len(class_bins)):
                    train.extend( subjects[label][:i*n_test] + subjects[label][(i+1)*n_test:] )
                    test.extend( subjects[label][i*n_test:(i+1)*n_test] )
                np.random.shuffle(test) # needs extra shuffle because otherwise all examples of the same class would be next to each other
                train, validation = self.split_set(train,
                                                split_size = 1 - self.validation_size,
                                                balance_first_set = self.balance_train_set,
                                                balance_second_set = True)
                self.data_distr[f"fold_{i+1}/{k}"] = data_distribution(train, validation, test)
                yield to_subject_datasets(train, validation, test)
        else:
            assert False, "k_fold: NOT IMPLEMENTED"
        
    def split_set(self, patients: list, split_size: float = 0.8, 
        balance_first_set: bool = True, balance_second_set: bool = True):
        '''
        TODO
        Splits the patients list into two sets. 
        When balance_first_set is true, the first dataset is balanced using data 
        augmentation.
        When balance_second_set is true, it is balanced by undersampling the 
        majority class (because this set represents the validation or test set).
        '''
        assert 0 < split_size < 1
        if balance_second_set:
            bins        = to_bins(patients)
            bin_count   = [len(bins[label]) for label in bins]
            min_class   = min(bin_count)
            max_class   = max(bin_count)
            n_first     = round(min_class * split_size)         # number of elements per class in the first set
            n_second    = min_class - n_first                   # number of elements per class in the second set
            n_augment   = n_first * (len(self.transforms)+1)    # number of elements per class after augmentations
            first_set   = []
            second_set  = []
            for label in bins:
                s1 = bins[label][n_second:]
                s2 = bins[label][:n_second]
                if (len(s1) < 1) or (len(s2) < 1):
                    print(f"WARNING: One of the sets will have no examples for class {label}. Choose splits closer to 0.5 to avoid this issue.")
                if balance_first_set:
                    first_set.extend( self.augment(s1, n_augment) )
                else:
                    first_set.extend( s1 )
                second_set.extend(s2)
            np.random.shuffle(first_set)
            np.random.shuffle(second_set)
            return first_set, second_set
        else:
            assert False, "split_set: NOT IMPLEMENTED"


if __name__ == "__main__":
    TABLE_DATA  = "gravo.csv"
    DATA_DIR    = "../../../data/gravo"
    ct_loader   = CTLoader(TABLE_DATA, "NCCT", 
                        balance_test_set    = True,
                        balance_train_set   = False,
                        data_dir            = DATA_DIR)
    
    train, validation, test = ct_loader.subject_dataset()
    test_loader = torch.utils.data.DataLoader(test, 
                            batch_size  = 2, 
                            num_workers = 0,
                            pin_memory  = torch.cuda.is_available())
