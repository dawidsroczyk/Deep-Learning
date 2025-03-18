import kagglehub
import os


def get_dataset_path():
    '''
    Function for downloading data from kaggle
    It returns path to the dataset on the disk
    '''
    path = kagglehub.dataset_download("mengcius/cinic10")
    return path


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