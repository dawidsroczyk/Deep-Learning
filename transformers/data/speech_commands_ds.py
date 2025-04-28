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
        
        # Define label mapping as per paper
        self.valid_labels = {'yes', 'no', 'up', 'down', 'left', 'right', 
                            'on', 'off', 'stop', 'go'}
        if not vsu:
            self.class_to_idx = {label: i for i, label in enumerate(sorted(self.valid_labels))}
        else:
            self.class_to_idx = {'valid': 0}
            
        self.class_to_idx['silence'] = len(self.class_to_idx)  # Add silence class
        self.class_to_idx['unknown'] = len(self.class_to_idx)  # Add unknown class
        
        # Load and filter files
        self.samples = []
        with open(os.path.join(root_dir, file_list_path), 'r') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:  # Skip empty lines
                    continue
                    
                # Extract label from folder name
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
        
        # Load audio
        waveform, sample_rate = torchaudio.load(full_path)
        
        # Resample if needed (paper uses 16kHz)
        if sample_rate != self.preprocessor.sample_rate:
            resampler = T.Resample(sample_rate, self.preprocessor.sample_rate)
            waveform = resampler(waveform)
        
        # Apply augmentations if training (as described in paper)
        if self.augment:
            waveform = self.preprocessor.apply_data_augmentation(waveform)
        
        # Process based on mode
        if self.mode == '1d':
            data = self.preprocessor.preprocess_waveform(waveform)
        else:  # '2d'
            data = self.preprocessor.preprocess_spectrogram(waveform)
        
        # Get class index
        class_idx = self.class_to_idx[label]
        
        return data, class_idx