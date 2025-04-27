import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import json
import random
import os
import torch.nn.functional as F
from torch.utils.data import random_split


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class AST(nn.Module):
    def __init__(self, num_classes, model_dim=768, patch_dim=256, dropout=0.0):
        super(AST, self).__init__()
        self.projection = nn.Linear(patch_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=12, dim_feedforward=4*model_dim, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.linear = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        x = self.projection(x)
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        cls_output = x[0]
        logits = self.linear(cls_output)
        return logits
    
    def check_unknown(self, x, threshold=0.5):
        '''
        return 1 in unknown, 0 otherwise
        '''
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            max_probs, _ = torch.max(probs, dim=1)
            result = (max_probs < threshold).int().cpu().numpy()
            return result

class DictDataset(Dataset):
    def __init__(self, dic, class_to_idx):
        self.data = []
        self.labels = []
        for key, data_key in dic.items():
            self.data.extend(data_key)
            self.labels.extend([class_to_idx[key]] * len(data_key))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_dataset(train_path, test_path):
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    class_to_idx = dict([(key, idx) for idx, key in enumerate(train_data.keys())])
    print(class_to_idx)
    train_dataset = DictDataset(train_data, class_to_idx)
    test_dataset = DictDataset(test_data, class_to_idx)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=4)
    return train_dataloader, test_dataloader

def load_unknown_dataset(unknown_path, max_elems: int = None):
    unknown_data = torch.load(unknown_path)
    unknown_dataset = DictDataset(unknown_data, {'unknown': 0})
    if max_elems is not None and max_elems < len(unknown_dataset):
        # unknown_dataset = unknown_dataset[:max_elems]
        total_size = len(unknown_dataset)
        unknown_dataset, _ = random_split(unknown_dataset, [max_elems, total_size-max_elems])
    unknown_dataloader = DataLoader(unknown_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
    return unknown_dataloader

@torch.no_grad()
def evaluate_with_unknown(model, test_dataloader, unknown_dataloader, device, num_classes, threshold=0.5):
    model.eval()
    all_preds = []
    all_true_labels = []
    
    # Process test samples (known classes)
    for seq, labels in test_dataloader:
        seq, labels = seq.to(device), labels.to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(seq)
                probs = torch.sigmoid(logits, dim=1)
                max_probs, preds = torch.max(probs)
                
                # Convert to numpy while preserving device sync
                preds = torch.where(max_probs < threshold, 
                                  num_classes, 
                                  preds).cpu().numpy()
                preds = preds.tolist()
                
        all_preds.extend(preds)
        all_true_labels.extend(labels.cpu().numpy().tolist())
    
    print(all_preds)

    # Process unknown samples
    for seq, _ in unknown_dataloader:
        seq = seq.to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(seq)
                probs = torch.sigmoid(logits)
                max_probs = probs.max(dim=1).values
                
                preds = (max_probs < threshold).int().cpu().numpy()
                preds = np.where(preds == 1, num_classes, 
                                torch.argmax(logits, dim=1).cpu().numpy())
                preds = preds.tolist()

        all_preds.extend(preds)
        all_true_labels.extend([num_classes] * len(seq))
    
    print(all_true_labels)

    # Ensure all labels are within expected range
    unique_labels = set(all_true_labels + all_preds)
    assert all(0 <= label <= num_classes for label in unique_labels), \
           f"Invalid labels detected: {unique_labels}"

    conf_matrix = confusion_matrix(
        all_true_labels,
        all_preds,
        labels=list(range(num_classes + 1))
    )
    
    return conf_matrix, accuracy_score(all_true_labels, all_preds)


def pad_dataloaders(dataloaders: list, batch_size: int):
    # Step 1: Collect all sequences from all dataloaders to find max length
    all_sequences = []
    for dataloader in dataloaders:
        for batch in dataloader:
            # Remove batch dim (original batch_size=1) to get [seq_len, 256]
            sequence = batch[0].squeeze(0)
            all_sequences.append(sequence)
    
    max_len = max(seq.size(0) for seq in all_sequences)  # Find max seq length
    
    # Step 2: Process each original dataloader separately
    padded_dataloaders = []
    for dataloader in dataloaders:
        sequences = []
        labels = []
        for batch in dataloader:
            sequence = batch[0].squeeze(0)  # [seq_len, 256]
            sequences.append(sequence)
            labels.append(batch[1])
        
        # Pad sequences in this dataloader to max_len
        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0
        )  # [num_sequences, max_len, 256]
        labels = torch.stack(labels)
        
        # Create new DataLoader with desired batch_size
        dataset = TensorDataset(padded_sequences, labels)
        new_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False  # Preserve original order (or set True if needed)
        )
        padded_dataloaders.append(new_dataloader)
    
    return tuple(padded_dataloaders)

def train_ast(num_epochs, num_classes, lr, weight_decay, model_path, 
              train_data_path, test_data_path, unknown_data_path, random_seed, confusion_matrices_path,
              unknown_threshold, batch_size):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = load_dataset(train_data_path, test_data_path)
    unknown_dataloader = load_unknown_dataset(unknown_data_path)
    train_dataloader, test_dataloader, unknown_dataloader = pad_dataloaders([train_dataloader, test_dataloader, unknown_dataloader], batch_size)

    model = AST(num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_test_acc = 0.0

    scaler = torch.cuda.amp.GradScaler() 
    for epoch in range(num_epochs):
        model.train()
        train_preds = []
        train_labels = []
        total_loss = 0

        for idx, (seq, labels) in enumerate(train_dataloader):
            if idx % 100 == 0:
                print(f'{idx} / {len(train_dataloader)}')
            seq, labels = seq.to(device), labels.to(device)
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float().squeeze(1)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(seq)
                loss = criterion(logits, one_hot_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_conf = confusion_matrix(train_labels, train_preds)

        test_conf, test_acc = evaluate_with_unknown(model, test_dataloader, unknown_dataloader, device, num_classes, unknown_threshold)

        torch.save(model.state_dict(), os.path.join(model_path, f'model_{epoch}.pt'))

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        print("Train Confusion Matrix:\n", train_conf)
        print("Test Confusion Matrix:\n", test_conf)
        print("-" * 50)

        conf_data = {
            "train_confusion_matrix": train_conf.tolist(),
            "test_confusion_matrix": test_conf.tolist()
        }

        with open(f"{confusion_matrices_path}/epoch_{epoch+1}_confusion_matrices.json", "w") as f:
            json.dump(conf_data, f, indent=4)

    print("✅ Training complete.")