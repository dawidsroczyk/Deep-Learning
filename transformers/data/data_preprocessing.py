import py7zr
import os
import numpy as np
import torchaudio
from sklearn.model_selection import train_test_split

def extract_data(directory):
    with py7zr.SevenZipFile(f'{directory}/train.7z', mode='r') as z:
        z.extractall(path=directory)

    os.remove(f'{directory}/train/audio/_background_noise_/README.md')

def get_train_list(directory):
    path = f'{directory}/train/audio'
    
    with open(f'{directory}/train/validation_list.txt') as file:
        validation = file.read().splitlines()

    with open(f'{directory}/train/testing_list.txt') as file:
        test = file.read().splitlines()
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for name in filenames:
            s = str(os.path.join(os.path.relpath(dirpath, path), name)).replace('\\', '/')
            if s not in validation and s not in test and not s.startswith('_background_noise_'):
                f.append(s)

    with open(f'{directory}/train/train_list.txt', 'w') as txt_file:
        for file in f:
            txt_file.write(f'{file}\n')

def _cut(sig, sr, ms):
    max_len = sr // 1000 * ms
    sig_len = sig.shape[1]
    count = sig_len // max_len
    split = np.hsplit(sig[:, :max_len * count], count)
    if sig_len > max_len * count:
      split.append(sig[:, max_len * count + 1:])
    return split, sr

def get_silence_files(directory, random_state = 42):
    path_target = f'{directory}/train/audio/silence'
    path_source = f'{directory}/train/audio/_background_noise_'
    os.mkdir(path_target)
    for file in os.listdir(path_source):
        sig, sr = torchaudio.load(os.path.join(path_source, file))
        split, sr = _cut(sig, sr, 1000)
        for i, sig in enumerate(split):
            torchaudio.save(os.path.join(path_target, file[:-4] + str(i) + '.wav'), sig, sr)
    
    train, temp = train_test_split(np.array(os.listdir(path_target)), test_size=0.2, random_state=random_state)
    validation, test = train_test_split(temp, test_size=0.5, random_state=random_state)

    with open(f'{directory}/train/train_list.txt', "a") as train_list:
        for file in train:
            train_list.write(f'silence/{file}\n')
            
    with open(f'{directory}/train/validation_list.txt', "a") as validation_list:
        for file in validation:
            validation_list.write(f'silence/{file}\n')

    with open(f'{directory}/train/testing_list.txt', "a") as test_list:
        for file in test:
            test_list.write(f'silence/{file}\n')

if __name__ == '__main__':
    directory = 'C:/Users/weron/Pulpit/sem1/dl/proj1/Deep-Learning/transformers/dataset'
    extract_data(directory)
    get_train_list(directory)
    get_silence_files(directory)