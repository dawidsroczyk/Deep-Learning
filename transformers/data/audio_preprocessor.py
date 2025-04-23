import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
from torch.utils.data import Dataset
import pandas as pd
import os

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_fft=1024, win_length=400, 
                 hop_length=160, n_mels=64, duration_ms=1024):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or win_length // 4
        self.n_mels = n_mels
        self.duration_ms = duration_ms
        self.target_length = int(sample_rate * duration_ms / 1000)
        
        # Mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window,  # Hamming window 
            n_mels=n_mels,
            power=2.0
        )
        
        # Amplitude to DB transform
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def preprocess_waveform(self, waveform):
        """Preprocess raw waveform for 1D CNN"""
        # Ensure correct length
        if waveform.shape[1] > self.target_length:
            # Random crop
            start = np.random.randint(0, waveform.shape[1] - self.target_length)
            waveform = waveform[:, start:start+self.target_length]
        elif waveform.shape[1] < self.target_length:
            # Pad with zeros
            pad_left = (self.target_length - waveform.shape[1]) // 2
            pad_right = self.target_length - waveform.shape[1] - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
        
        # Standardize volume
        waveform = self.normalize_volume(waveform)
        return waveform

    def preprocess_spectrogram(self, waveform):
        """Preprocess waveform into log-Mel spectrogram for 2D CNN"""
        # Process waveform to correct length
        waveform = self.preprocess_waveform(waveform)
        
        # Convert to Mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to dB scale
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Add channel dimension (for CNN)
        log_mel_spec = log_mel_spec.unsqueeze(0)
        
        return log_mel_spec

    def normalize_volume(self, waveform, target_dBFS=-30):
        """Normalize audio volume to target dBFS"""
        rms = torch.sqrt(torch.mean(waveform**2))
        target_rms = 10 ** (target_dBFS / 20)
        waveform = waveform * (target_rms / (rms + 1e-6))
        return torch.clamp(waveform, -1.0, 1.0)

    def apply_data_augmentation(self, waveform):
        """Apply data augmentation as described in the paper"""
        # Time stretching (0.7-1.4 rate)
        rate = np.random.uniform(0.7, 1.4)
        waveform = torchaudio.functional.resample(
            waveform, 
            orig_freq=int(self.sample_rate), 
            new_freq=int(self.sample_rate * rate))
        
        # Time shifting (-0.1 to +0.1 seconds)
        shift = np.random.randint(-int(0.1 * self.sample_rate), 
                               int(0.1 * self.sample_rate))
        waveform = torch.roll(waveform, shifts=shift, dims=1)
        
        # Add random background noise (0-5% of max volume)
        max_vol = torch.max(torch.abs(waveform))
        noise_level = np.random.uniform(0, 0.05) * max_vol
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
        
        return waveform