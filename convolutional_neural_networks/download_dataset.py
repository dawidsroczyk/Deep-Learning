import kagglehub
import os
import shutil
from torchvision import transforms
import torchvision


def get_dataset_path(augmented=False, n_augmented=3):
    '''
    Function for downloading data from kaggle
    It returns path to the dataset on the disk
    '''
    if augmented:
        raw_path = get_dataset_path(augmented=False)
        save_path = os.path.join('data', 'augmented')
        save_path_train = os.path.join(save_path, 'train')
        save_path_test = os.path.join(save_path, 'test')
        os.makedirs(save_path_train, exist_ok=True)
        os.makedirs(save_path_test, exist_ok=True)
        if len(os.listdir(save_path_train)) > 0:
            return save_path
        
        # save unchanged test files
        raw_path_test = os.path.join(raw_path, 'test')
        shutil.copytree(raw_path_test, save_path_test, dirs_exist_ok=True)
        
        augment_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        raw_data_train = torchvision.datasets.ImageFolder(os.path.join(raw_path, 'train'))
        for idx, (image, label) in enumerate(raw_data_train):
            class_name = raw_data_train.classes[label]
            class_save_path = os.path.join(save_path, 'train', class_name)
            os.makedirs(class_save_path, exist_ok=True)

            # Save the original image
            original_save_path = os.path.join(class_save_path, f"original_{idx}.jpg")
            image.save(original_save_path)

            # Generate and save augmented images
            for aug_idx in range(n_augmented):
                augmented_image = augment_transform(image)
                augmented_image = transforms.ToPILImage()(augmented_image)
                augmented_save_path = os.path.join(class_save_path, f"augmented_{idx}_{aug_idx}.jpg")
                augmented_image.save(augmented_save_path)
        
        return save_path

    else:
        save_path = os.path.join('data', 'raw')
        os.makedirs(save_path, exist_ok=True)
        if len(os.listdir(save_path)) > 0:
            return save_path
        path = kagglehub.dataset_download("mengcius/cinic10")
        for item in os.listdir(path):
            shutil.move(os.path.join(path, item), save_path)
        return save_path


def get_train_dataset_path(augmented=False):
    '''
    Get path to the saved train dataset
    '''
    root_path = get_dataset_path(augmented)
    train_path = os.path.join(root_path, 'train')
    return train_path


def get_test_dataset_path(augmented=False):
    '''
    Get path to the saved test dataset
    '''
    root_path = get_dataset_path(augmented)
    test_path = os.path.join(root_path, 'test')
    return test_path


if __name__ == '__main__':
    path = get_dataset_path()
    print("Path to dataset files:", path)