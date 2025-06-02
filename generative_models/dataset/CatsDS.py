import numpy as np
from torch.utils.data import Dataset
from PIL import Image
#import imageio
import os

class CatsDS(Dataset):

    def __init__(self, data_path, transform): 
        self.data_path = data_path
        self.transform = transform
        self.files = [
            f for f in os.listdir(data_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_full = os.path.join(self.data_path, file)

        #image = Image.fromarray(imageio.imread(file_full))
        image = Image.open(file_full).convert('RGB') 

        if self.transform:
            return [self.transform(image)]
        
        return [image]