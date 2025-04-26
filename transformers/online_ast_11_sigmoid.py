import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import json
import random
import os
import torch.nn.functional as F


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
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    return train_dataloader, test_dataloader

def load_unknown_dataset(unknown_path):
    unknown_data = torch.load(unknown_path)
    unknown_dataset = DictDataset(unknown_data)
    unknown_dataloader = DataLoader(unknown_dataset, batch_size=1)
    return unknown_dataloader

def evaluate_with_unknown(model, test_dataloader, unknown_dataloader, device, num_classes, threshold=0.5):
    model.eval()
    all_preds = []
    all_true_labels = []
    
    with torch.no_grad():
        # 1. Process known test samples
        for seq, labels in test_dataloader:
            seq, labels = seq.to(device), labels.to(device)
            
            # Stage 1: Unknown detection (returns array)
            is_unknown_arr = model.check_unknown(seq, threshold)
            
            # Stage 2: Classification for non-unknown samples
            logits = model(seq)
            class_preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Combine results (use num_classes where unknown)
            batch_preds = np.where(is_unknown_arr == 1, num_classes, class_preds)
            
            all_preds.extend(batch_preds)
            all_true_labels.extend(labels.cpu().numpy())
        
        # 2. Process unknown samples
        for seq, _ in unknown_dataloader:
            seq = seq.to(device)
            is_unknown_arr = model.check_unknown(seq, threshold)
            
            # True label is unknown (num_classes)
            batch_true_labels = np.full(len(seq), num_classes)
            
            # Predictions should be unknown (num_classes)
            batch_preds = np.where(is_unknown_arr == 1, num_classes, 
                                 torch.argmax(model(seq), dim=1).cpu().numpy())
            
            all_preds.extend(batch_preds)
            all_true_labels.extend(batch_true_labels)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(
        all_true_labels,
        all_preds,
        labels=list(range(num_classes + 1))
    )
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_preds)
    unknown_recall = conf_matrix[-1,-1] / conf_matrix[-1,:].sum()
    false_unknown = conf_matrix[:-1,-1].sum() / conf_matrix[:-1,:].sum()
    
    return conf_matrix, accuracy

def train_ast(num_epochs, num_classes, lr, weight_decay, model_path, 
              train_data_path, test_data_path, unknown_data_path, random_seed, confusion_matrices_path,
              unknown_threshold):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader = load_dataset(train_data_path, test_data_path)
    unknown_dataloader = load_unknown_dataset(unknown_data_path)
    model = AST(num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_preds = []
        train_labels = []
        total_loss = 0
        for idx, (seq, labels) in enumerate(train_dataloader):
            if idx % 1000 == 0:
                print(f'{idx} / {len(train_dataloader)}')
            seq, labels = seq.to(device), labels.to(device)
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
            optimizer.zero_grad()
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