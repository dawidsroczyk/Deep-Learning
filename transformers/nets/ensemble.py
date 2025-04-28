import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class Ensemble(nn.Module):
    def __init__(self, general_models, vsu_model):
        """
        Ensemble model that combines VSU classification with general models.
        
        Args:
            general_models: List of models trained on all classes (valid + silence + unknown)
            vsu_model: Model trained specifically for valid/silence/unknown classification
        """
        super(Ensemble, self).__init__()
        self.general_models = nn.ModuleList(general_models)
        self.vsu_model = vsu_model
        
        self.class_to_idx = {
            'down': 0, 'go': 1, 'left': 2, 'no': 3, 'off': 4, 
            'on': 5, 'right': 6, 'silence': 7, 'stop': 8, 'up': 9, 'yes': 10,
            'unknown': 11
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.vsu_class_to_idx = {'valid': 0, 'silence': 1, 'unknown': 2}
        self.vsu_idx_to_class = {v: k for k, v in self.vsu_class_to_idx.items()}
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # First get VSU predictions for the whole batch
        vsu_output = self.vsu_model(x)
        _, vsu_preds = torch.max(vsu_output, 1)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, len(self.class_to_idx)).to(device)
        
        # Process each sample in the batch
        for i in range(batch_size):
            vsu_class_idx = vsu_preds[i].item()
            vsu_class = self.vsu_idx_to_class[vsu_class_idx]
            
            if vsu_class == 'valid':
                # Get predictions from all general models for this sample
                sample_preds = []
                for model in self.general_models:
                    output = model(x[i].unsqueeze(0))  # Process single sample
                    _, pred = torch.max(output, 1)
                    sample_preds.append(pred.item())
                
                # Majority voting
                majority_class_idx = Counter(sample_preds).most_common(1)[0][0]
                final_class = self.idx_to_class[majority_class_idx]
                final_idx = self.class_to_idx[final_class]
                
            else:  # silence or unknown
                final_idx = self.class_to_idx[vsu_class]
            
            # Set the appropriate output index
            outputs[i, final_idx] = 1.0
        
        return outputs