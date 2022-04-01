import os, torch, numpy
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def load_encodings_dataset(dir: str):
    encodings_dir = f"{dir}/encodings"
    assert os.path.isdir(encodings_dir), "load_encodings: No encodings dir folder found. Run 'Trainer.save_encodings' first"
    out = []
    for set_ in ("train", "validation", "test"):
        dir         = f"{encodings_dir}/{set_}"
        subjects    = []
        i           = 0
        while True:
            to_load = f"{dir}/subject-{i}.pt"
            if not os.path.isfile(to_load):
                break
            subject = torch.load(to_load, map_location = torch.device("cpu"))
            subjects.append( subject.detach().numpy() )
            i += 1
        subjects = numpy.stack(subjects)
        y = torch.load(f"{dir}/labels.pt", map_location = torch.device("cpu"))
        assert len(subjects) == len(y), f"load_encodings_dataset: number of examples and labels doesn't match for {set_}"
        out.append( (subjects, y) )
    return out
    
def get_train_test_sets(dir: str):
    train, val, test = load_encodings_dataset(dir)
    x_train, y_train = train
    x_val, y_val     = val
    x_train          = numpy.concatenate((x_train, x_val), axis = 0)
    y_train          = numpy.concatenate((y_train, y_val), axis = 0)
    return (x_train, y_train), test
    
    
if __name__ == "__main__":
    # adapted from
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    train, test = get_train_test_sets("SiameseNet-8.31.")
    x_train, y_train = train
    x_test, y_test   = test
    svc_parameters = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]
    knn_parameters = [
        {"n_neighbors": [3, 5, 7, 9], "metric": ["euclidean", "manhattan", "chebyshev"]}
    ]
    clf = GridSearchCV(KNN(), knn_parameters, scoring="accuracy")
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
