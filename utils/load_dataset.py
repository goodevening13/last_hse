import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    def __init__(self, texts, labels, augment=False, extra=False):
        if extra:
            self.texts = [elem[0] for elem in texts]
            self.attn = [elem[1] for elem in texts]
        else:
            self.texts = texts.to_list()
        self.labels = labels.to_list()
        self.extra = extra
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = augmentations(text)
        if self.extra:
            return torch.LongTensor(text), self.labels[idx], self.attn[idx]
        return torch.LongTensor(text), self.labels[idx], None
    

def augmentations(embs, p=0.5):
    for _ in range(5):
        k, l = [random.randint(1, len(embs) - 1) for i in range(2)]
        embs[k], embs[l] = embs[l], embs[k]
    if random.random() < p:
        embs = [embs[0]] + [w for w in embs[1:-1] if random.random() > 0.1] + [embs[-1]]
    return embs


def collate_fn(batch, pad_token_id=128004):
    sequences, labels, extra = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    labels = torch.LongTensor(labels)
    lengths = torch.tensor([len(seq) for seq in sequences])
    if extra:
        return padded_sequences, labels, extra, lengths
    return padded_sequences, labels, None, lengths


def create_loaders(train_ds, val_ds, test_ds, batch_size, embs_column='embs', extra=False):
    if extra:
        train_dataset = TextDataset(train_ds[embs_column], train_ds['enc_label'], augment=True, extra=extra)
        val_dataset = TextDataset(val_ds[embs_column], val_ds['enc_label'], augment=False, extra=extra)
        test_dataset = TextDataset(test_ds[embs_column], test_ds['enc_label'], augment=False, extra=extra)
    else:
        train_dataset = TextDataset(train_ds[embs_column], train_ds['enc_label'], augment=True)
        val_dataset = TextDataset(val_ds[embs_column], val_ds['enc_label'], augment=False)
        test_dataset = TextDataset(test_ds[embs_column], test_ds['enc_label'], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
