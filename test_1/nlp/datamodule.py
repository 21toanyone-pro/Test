import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.datasets import AG_NEWS
from transformers import RobertaTokenizer
import pytorch_lightning as pl

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'].tolist(), truncation=True, padding=True, max_length=128)

class AGNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class AGNewsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

    def setup(self, stage=None):
        # 데이터셋 로드
        train_iter, test_iter = AG_NEWS()
        train_data = [(label, text) for (label, text) in train_iter]
        test_data = [(label, text) for (label, text) in test_iter]

        train_df = pd.DataFrame(train_data, columns=["label", "text"])
        test_df = pd.DataFrame(test_data, columns=["label", "text"])

        # 레이블
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1

        train_encodings = preprocess_function(train_df, self.tokenizer)
        test_encodings = preprocess_function(test_df, self.tokenizer)

        train_labels = torch.tensor(train_df['label'].values)
        test_labels = torch.tensor(test_df['label'].values)

        self.train_dataset = AGNewsDataset(train_encodings, train_labels)
        self.test_dataset = AGNewsDataset(test_encodings, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)