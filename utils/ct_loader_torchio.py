import torch, torchio, os
import pandas as pd
import numpy as np

augmentations = [torchio.RandomAffine(scales = 0, translation = 0, degrees = 10, center = "image"),
                 torchio.RandomElasticDeformation(),
                 torchio.RandomNoise(mean = 5, std = 2)]
rescale       =  torchio.RescaleIntensity(out_min_max = (0, 1))


def to_subject_datasets(train, validation, test):
    '''
    Input:  train, list of torhcio subjects objects
            validation, list of torhcio subjects objects
            test, list of torhcio subjects objects
    Output: 3 torchio SubjectsDataset created from the 3 inputed lists. Their 
            scans are rescaled so every voxel is between 0 and 1
    '''
    return  torchio.SubjectsDataset(train, transform = rescale),\
            torchio.SubjectsDataset(validation, transform = rescale),\
            torchio.SubjectsDataset(test, transform = rescale)


def list_scans(path: str = None):
    '''
    Input:  path, a string with a path to a directory
    Output: list of file names inside the path given as argument, whose file extension
            is ".nii" (i.e. the NIfTI format)
    '''
    return [c for c in os.listdir(path) if c.endswith(".nii")]


def binary(label: int):
    '''
    Behaviour:  Applies the binarization function proposed in "Using Machine Learning to 
                Improve the Prediction of Functional Outcome in Ischemic Stroke Patients", 
                for the target variable (mRS).
    Input:      label, a integer
    Output:     integer (0 or 1)
    '''
    return 0 if label <= 2 else 1
    
    
def to_bins(train: list):
    '''
    Input:  train, list of torchio subjects
    Output: a dictionary whose keys are the different classes and its values are 
            the list of torchio subjects that belong to that class
    '''
    bins = {}
    for t in train:
        if t["prognosis"] in bins:
            bins[t["prognosis"]].append(t)
        else:
            bins[t["prognosis"]] = [t]
    return bins


def to_bin_count(patients: list):
    '''
    Input:  patients, list of torchio subjects
    Output: a dictionary whose keys are the different classes and its values are 
            the len of the list of torchio subjects that belong to that class
    '''
    bins = {}
    for t in patients:
        if t["prognosis"] in bins:
            bins[t["prognosis"]] += 1
        else:
            bins[t["prognosis"]] = 1
    return bins
    
    
def augment(to_augment, n_augment):
    bin_size    = len(to_augment)
    to_add      = n_augment - bin_size
    flip        = torchio.RandomFlip("lr")
    for i in range(min(to_add, bin_size)):
        to_augment.append( flip(to_augment[i]) )
    to_add -= bin_size
    for i in range(to_add):
        # i//bin_size - cycle through the to_augment set
        # %len(augmentations) - cycle through the augmentations
        to_augment.append( augmentations[(i//bin_size)%len(augmentations)](to_augment[i%bin_size]) )
    return to_augment
    

def data_distribution(train, validation, test):
    '''
    Input:  train, list of torhcio subjects objects
            validation, list of torhcio subjects objects
            test, list of torhcio subjects objects
    Output: a dictionary whose keys are the strings 'train', 'validation' and 'test'
            and values are the respective data data distributions of their lists
    '''
    distr               = {}
    distr["train"]      = to_bin_count(train)
    distr["validation"] = to_bin_count(validation)
    distr["test"]       = to_bin_count(test)
    return distr
    

class CT_loader:
    def __init__(self, table_data_file: str, ct_type: str, binary_label: bool = True, 
        has_both_scan_types: bool = False, balance_test_set: bool = True, 
        random_seed: int = None, balance_train_set: bool = False, 
        augment_factor: float = 1., data_dir: str = None, validation_size: float = 0.2):
        '''
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
                augment_factor, a float higher than or equal to 1 specifying the increase
                factor in the train set
                data_dir, string with the path location of the ct_type directory and table_data_file
                validation_size, float specifying the relative size of the validation set (this percentage
                is relative to the size of the trai set, not the whole dataset)
        '''
        assert ct_type in ("NCCT","CTA"), f"CT_loader.__init__: ct_type must be NCCT or CTA"
        assert 0 < validation_size < 1, "CT_loader.__init__: validation_size must be between 0 and 1"
        assert augment_factor >= 1, "CT_loader.__init__: can't decrease train set size"
        self.table_data_file        = table_data_file
        self.ct_type                = ct_type
        self.binary_label           = binary_label
        self.has_both_scan_types    = has_both_scan_types
        self.balance_test_set       = balance_test_set
        self.random_seed            = random_seed
        self.balance_train_set      = balance_train_set
        self.augment_factor         = augment_factor
        self.data_dir               = data_dir
        self.validation_size        = validation_size
        self.data_distr             = {}
        self.init_table_data()
        
    def create_subject(self, row):
        '''
        Input:  row, pandas DataFrame row of the patient to be loaded into a subject torchio object
        Output: torchio object with the loaded scan
        '''
        if self.data_dir is not None:
            ct_type = os.path.join(self.data_dir, self.ct_type)
        label   = int(row["rankin-3m"])
        scan    = row["idProcessoLocal"]
        subject = torchio.Subject(
                ct  = torchio.ScalarImage(os.path.join(ct_type,f"{scan}.nii")),
                prognosis = label)
        return subject
        
    def init_table_data(self):
        '''
        Output: pandas DataFrame with the table_data_file csv loaded that satisfies the
                inputted settings
        '''
        if self.data_dir is not None:
            self.table_data_file = os.path.join(self.data_dir, self.table_data_file)
        table_data = pd.read_csv(self.table_data_file)
        table_data = table_data[table_data["rankin-3m"] != "missing"]
        table_data = table_data[table_data[self.ct_type] != "missing"]
        if self.has_both_scan_types:
            table_data = table_data[(table_data["NCCT"] != "missing") & (table_data["CTA"] != "missing")]
        if self.binary_label:
            table_data["rankin-3m"] = [binary(int(t)) for t in table_data["rankin-3m"].values]
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        random_permutation  = np.random.permutation( len(table_data) )
        table_data          = table_data.iloc[random_permutation]
        self.table_data     = table_data
        
    def to_dict(self):
        params  = ["ct_type", "binary_label", "has_both_scan_types",
                "balance_test_set", "random_seed", "balance_train_set",
                "augment_factor", "validation_size"]
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
        '''
        assert k > 1, "k_fold: for k=1 use create_subject_dataset with train_size = 0.5 instead"
        if self.balance_test_set:
            class_bins  = np.bincount( self.table_data["rankin-3m"] )
            min_class   = np.min(class_bins)
            n_test      = min_class // k
            subjects    = {label:[] for label in range(len(class_bins))}
            for _, row in self.table_data.iterrows():
                subjects[int(row["rankin-3m"])].append( self.create_subject(row) )
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
        
    def split_set(self, patients, split_size = 0.8, balance_first_set = True, balance_second_set = True):
        '''
        TODO
        '''
        assert 0 < split_size < 1
        if balance_second_set:
            bins        = to_bins(patients)
            bin_count   = [len(bins[label]) for label in bins]
            min_class   = min(bin_count)
            max_class   = max(bin_count)
            n_second    = round(min_class * (1 - split_size))               # number of elements per class in the second set
            n_first     = round((max_class-n_second) * self.augment_factor) # number of elements per class in the first set
            first_set   = []
            second_set  = []
            for label in bins:
                s2 = bins[label][:n_second]
                if (len(s2) < 1) or (len(bins[label][n_second:]) < 1):
                    print(f"WARNING: One of the sets will have no examples for class {label}. Choose splits closer to 0.5 to avoid this issue.")
                second_set.extend(s2)
                if balance_first_set:
                    first_set.extend( augment(bins[label][n_second:], n_first) )
                else:
                    first_set.extend( bins[label][n_second:] )
            np.random.shuffle(first_set)
            np.random.shuffle(second_set)
            return first_set, second_set
        else:
            assert False, "split_set: NOT IMPLEMENTED"
