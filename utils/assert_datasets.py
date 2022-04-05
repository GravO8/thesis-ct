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
        train_ids       = [subject["patient_id"] for subject in self.train]
        validation_ids  = [subject["patient_id"] for subject in self.validation]
        for patient_id in validation_ids:
            if patient_id in train_ids:
                print(f"WARNING! Patient {patient_id} is present in the train and validation sets.")
                leaks += 1
        for subject in self.test:
            patient_id = subject["patient_id"]
            if patient_id in train_ids:
                print(f"WARNING! Patient {patient_id} is present in the train and test sets.")
                leaks += 1
            if patient_id in validation_ids:
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
            for subject in dataset:
                patient_id  = subject["patient_id"]
                name        = patient_id + "-" + subject["transform"]
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
                for subject in dataset:
                    label = subject["target"]
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
                if not aux(self.train):
                    print(f"WARNING! Train dataset is not balanced.")
                if not aux(self.validation):
                    print(f"WARNING! Validation dataset is not balanced.")
                if not aux(self.test):
                    print(f"WARNING! Test dataset is not balanced.")
            else:
                if dataset_name == "train":
                    if not aux(self.train):
                        print(f"WARNING! Train dataset is not balanced.")
                elif dataset_name == "validation":
                    if not aux(self.validation):
                        print(f"WARNING! Validation dataset is not balanced.")
                elif dataset_name == "test":
                    if not aux(self.test):
                        print(f"WARNING! Test dataset is not balanced.")
                else:
                    assert False, f"AssertDatasets.assert_repeated: unknown dataset {dataset_name}"
