import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from config import *
from preprocess import *


def create_sequences(X, y, sequence_length, test=False):
    sequences = []
    data_size = len(X)

    for i in range(data_size - sequence_length):
        sequence = X[i: i + sequence_length]
        label = y[i + sequence_length]
        sequences.append((sequence, label))

    if test:
        sequence = X[data_size - sequence_length]
        label = y[-1]
        sequences.append((sequence, label))

    return sequences


class AQDataset(Dataset):
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        sequence, label = self.sequences[item]
        return dict(
            sequence=torch.Tensor(sequence).float(),
            label=torch.as_tensor(label).float()
        )


class AQDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, val_sequences, batch_size=8):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        print("fdsfsdfdsfgregfdghbrfhrghrtgtred" if self.train_sequences else 0)
        self.train_dataset = AQDataset(self.train_sequences)
        self.val_dataset = AQDataset(self.val_sequences)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=1)



def prepare_dataset(X_train, X_val, y_train, y_val):
    # Get the sequence
    train_sequences = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    val_sequences = create_sequences(X_val, y_val, SEQUENCE_LENGTH)

    data_module = AQDataModule(train_sequences, val_sequences, BATCH_SIZE)
    data_module.setup()

    return data_module


if __name__ == "__main__":
    # Read all data
    df_path_list = os.listdir("./data")
    df_list = [pd.read_csv(f"./data/{path}") for path in df_path_list]

    # Preprocess the data
    print("Preprocess the data")
    X_train, X_val, y_train, y_val = preprocess_data(df_list)

    train_sequences = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    print(train_sequences[0][0])
    print(train_sequences[0][1])

    train_dataset = AQDataset(train_sequences)

    for item in train_dataset:
        print(item["sequence"].shape)
        print(item["label"].shape)
        print(item["label"])

        break


