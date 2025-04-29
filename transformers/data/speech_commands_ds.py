from torch.utils.data import Dataset
import torchaudio
import os

class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, file_list_path, mode='1d', vsu=False,
                 preprocessor=None, augment=False):
        """
        Args:
            root_dir: Root directory of dataset
            file_list_path: Path to file containing list of audio files (one per line)
            mode: '1d' or '2d'
            vsu: if True only 'valid', 'silence' and 'unknown' classes
            preprocessor: AudioPreprocessor instance
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.mode = mode
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.augment = augment
        
        self.valid_labels = {'yes', 'no', 'up', 'down', 'left', 'right', 
                            'on', 'off', 'stop', 'go'}
        if not vsu:
            labels = {'yes', 'no', 'up', 'down', 'left', 'right', 
                            'on', 'off', 'stop', 'go', 'silence'}
            self.class_to_idx = {label: i for i, label in enumerate(sorted(labels))}
            self.class_to_idx['unknown'] = len(self.class_to_idx)  
        else:
            self.class_to_idx = {'valid': 0, 'silence': 1, 'unknown': 2}
                    
        self.samples = []
        with open(os.path.join(root_dir, file_list_path), 'r') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                    
                label = rel_path.split('/')[0]
                
                if label not in self.valid_labels and label != 'silence':
                    label = 'unknown'
                
                if label in self.valid_labels:
                    if vsu:
                        label = 'valid'
                
                full_path = os.path.join(os.path.join(root_dir, 'audio'), rel_path)
                if os.path.exists(full_path):
                    self.samples.append((rel_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(os.path.join(self.root_dir, 'audio'), rel_path)
        
        waveform, sample_rate = torchaudio.load(full_path)
        
        if sample_rate != self.preprocessor.sample_rate:
            resampler = T.Resample(sample_rate, self.preprocessor.sample_rate)
            waveform = resampler(waveform)
        
        if self.augment:
            waveform = self.preprocessor.apply_data_augmentation(waveform)
        
        if self.mode == '1d':
            data = self.preprocessor.preprocess_waveform(waveform)
        else:  # '2d'
            data = self.preprocessor.preprocess_spectrogram(waveform)
        
        class_idx = self.class_to_idx[label]
        
        return data, class_idx