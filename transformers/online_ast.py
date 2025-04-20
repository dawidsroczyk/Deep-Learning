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

def load_dataset(pt_path):
    data = torch.load(pt_path)
    class_to_idx = dict([(key, idx) for idx, key in enumerate(data.keys())])
    dataset = DictDataset(data, class_to_idx)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    return train_dataloader, test_dataloader

def train_ast(num_epochs, num_classes, lr, weight_decay, best_model_path, data_path, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader = load_dataset(data_path)
    model = AST(num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_conf_matrices = []
    test_conf_matrices = []
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
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_conf = confusion_matrix(train_labels, train_preds)
        train_conf_matrices.append(train_conf.tolist())

        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for seq, labels in test_dataloader:
                seq, labels = seq.to(device), labels.to(device)
                logits = model(seq)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_conf = confusion_matrix(test_labels, test_preds)
        test_conf_matrices.append(test_conf.tolist())

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✔️ New best model saved with test accuracy: {test_acc:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        print("Train Confusion Matrix:\n", train_conf)
        print("Test Confusion Matrix:\n", test_conf)
        print("-" * 50)

    conf_data = {
        "train_confusion_matrices": train_conf_matrices,
        "test_confusion_matrices": test_conf_matrices
    }

    with open("confusion_matrices.json", "w") as f:
        json.dump(conf_data, f, indent=4)

    print("✅ Training complete. Confusion matrices saved to confusion_matrices.json.")
    print(f"🏅 Best model saved as {best_model_path} with accuracy {best_test_acc:.4f}")

