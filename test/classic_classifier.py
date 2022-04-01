import os, torch, numpy


def load_encondings_dataset(dir: str):
    encodings_dir = f"{dir}/encodings"
    assert not os.path.isdir(encodings_dir), "load_encodings: No encodings dir folder found. Run 'Trainer.save_encodings' first"
    out = []
    for dir in [f"encodings_dir/{s}" for s in ("train", "validation", "test")]:
        subjects_filenames = [file for file in os.listdir(encodings_dir) if file.startswith("subject-") and file.endswith(".pt")]
        subjects = []
        for i in range(len(subjects_filenames)):
            subject = torch.load(f"{encodings_dir}/subject-{i}.pt")
            subjects.append( subject.numpy() )
        subjects = numpy.stack(subjects)
        y = torch.load(f"{encodings_dir}/labels.pt")
        out.append( (subjects, y) )
    return out
    
def get_train_test_sets(dir: str):
    train, val, test = load_encondings_dataset(dir)
    x_train, y_train = train
    x_val, y_val     = val
    x_train          = numpy.stack(x_train, x_val)
    y_train          = numpy.stack(y_train, y_val)
    return (x_train, y_train), test
    
    
if __name__ == "__main__":
    train, test = get_train_test_sets("SiameseNet-8.31.")
    
