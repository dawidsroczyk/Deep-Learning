import os
import kagglehub

def full_dataset_path_local():
    path = '/home/dawid/repos/Deep-Learning/generative_models/data/cats/Data/'
    return path

def full_dataset_path_kaggle():
    path = kagglehub.dataset_download("borhanitrash/cat-dataset")
    path = os.path.join(path, 'cats', 'Data')
    return path
