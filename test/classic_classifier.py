import os, torch
import numpy as np


def load_encodings(dir: str):
    encodings_dir = f"{dir}/encodings"
    assert not os.path.isdir(encodings_dir), "load_encodings: No encodings dir folder found. Run 'Trainer.save_encodings' first"
    subjects = [file for file in os.listdir(encodings_dir) if file.startswith("subject-") and file.endswith(".pt")]
    for i in range(len(subjects)):
        subject = torch.load
            
