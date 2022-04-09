class AssertDatasets:
    def __init__(self, train, validation, test):
        self.train      = train
        self.validation = validation
        self.test       = test
        
    def assert_leaks(self):
        '''
        TODO
        '''
        leaks           = 0
        train_ids       = [int(id) for batch in self.train      for id in batch["patient_id"]]
        validation_ids  = [int(id) for batch in self.validation for id in batch["patient_id"]]
        test_ids        = [int(id) for batch in self.test       for id in batch["patient_id"]]
        for patient_id in validation_ids:
            if patient_id in train_ids:
                print(f"WARNING! Patient {patient_id} is present in the train and validation sets.")
                leaks += 1
        for id in test_ids:
            if id in train_ids:
                print(f"WARNING! Patient {patient_id} is present in the train and test sets.")
                leaks += 1
            if id in validation_ids:
                print(f"WARNING! Patient {patient_id} is present in the validation and test sets.")
                leaks += 1
        print(f"Found {leaks} leaks between train, validation and test sets.")
        
    def assert_repeated(self, dataset_name = "all"):
        '''
        TODO
        '''
        def aux(dataset, dataset_name):
            seen        = []
            repeated    = 0
            for batch in dataset:
                ids         = [str(id) for id in batch["patient_id"]]
                transforms  = [str(t) for t in  batch["transform"]]
                for i in range(len(ids)):
                    patient_id  = ids[i]
                    name        = patient_id + "-" + transforms[i]
                    if name in seen:
                        print(f"WARNING! Example {patient_id} repeated in dataset {dataset_name}")
                        repeated += 1
                    seen.append( name )
            print(f"Found {repeated} examples repeated in the {dataset_name} dataset.")
        if dataset_name == "all":
            aux(self.train, "train")
            aux(self.validation, "validation")
            aux(self.test, "test")
        else:
            if dataset_name == "train":
                aux(self.train, "train")
            elif dataset_name == "validation":
                aux(self.validation, "validation")
            elif dataset_name == "test":
                aux(self.test, "test")
            else:
                assert False, f"AssertDatasets.assert_repeated: unknown dataset {dataset_name}"
        
    def assert_balanced(self, dataset_name = "all"):
        '''
        TODO
        '''
        def aux(dataset):
            classes = {}
            for batch in dataset:
                for label in batch["target"]:
                    label = int(label)
                    if label in classes:
                        classes[label] += 1
                    else:
                        classes[label] = 1
            classes = [classes[label] for label in classes]
            for i in range(len(classes)-1):
                if classes[i] != classes[i+1]:
                    return False
            return True
        if dataset_name == "all":
            if aux(self.train):
                print("Train set is balanced.")
            else:
                print(f"WARNING! Train dataset is not balanced.")
            if aux(self.validation):
                print("Validation set is balanced.")
            else:
                print(f"WARNING! Validation dataset is not balanced.")
            if aux(self.test):
                print("Test set is balanced.")
            else:
                print(f"WARNING! Test dataset is not balanced.")
        else:
            if dataset_name == "train":
                if aux(self.train):
                    print("Train set is balanced.")
                else:
                    print(f"WARNING! Train dataset is not balanced.")
            elif dataset_name == "validation":
                if aux(self.validation):
                    print("Validation set is balanced.")
                else:
                    print(f"WARNING! Validation dataset is not balanced.")
            elif dataset_name == "test":
                if aux(self.test):
                    print("Test set is balanced.")
                else:
                    print(f"WARNING! Test dataset is not balanced.")
            else:
                assert False, f"AssertDatasets.assert_repeated: unknown dataset {dataset_name}"
