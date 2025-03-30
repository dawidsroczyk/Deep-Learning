import kagglehub
import os
import shutil
from torchvision import transforms
import torchvision


def get_dataset_path():
    save_path = os.path.join('data', 'raw', 'imagenet-mini')
    os.makedirs(save_path, exist_ok=True)
    if len(os.listdir(save_path)) > 0:
        return save_path
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), save_path)
    return save_path


def get_train_dataset_path():
    '''
    Get path to the saved train dataset
    '''
    root_path = get_dataset_path()
    train_path = os.path.join(root_path, 'train')
    return train_path


def get_test_dataset_path():
    '''
    Get path to the saved test dataset
    '''
    root_path = get_dataset_path()
    test_path = os.path.join(root_path, 'test')
    return test_path


if __name__ == '__main__':
    path = get_dataset_path()
    print("Path to dataset files:", path)