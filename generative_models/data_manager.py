import os
import kagglehub
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def full_dataset_path_local():
    path = '/home/dawid/repos/Deep-Learning/generative_models/data/cats/Data/'
    return path

def full_dataset_path_kaggle():
    path = kagglehub.dataset_download("borhanitrash/cat-dataset")
    path = os.path.join(path, 'cats', 'Data')
    return path

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        
        if self.transform:
            img = self.transform(img)
        
        return img

def create_full_dataset_dataloader(path, batch_size, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] range
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = ImageFolderDataset(
        folder_path=path,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return dataloader