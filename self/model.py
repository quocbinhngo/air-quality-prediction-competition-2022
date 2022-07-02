import numpy as np
from sklearn.metrics import mean_squared_error
import torch.cuda
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from config import *
from dataset import *


class AQPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(AQPredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=0.2)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        x = self.regressor(hidden[-1])
        return x


class AQPredictor(pl.LightningModule):
    def __init__(self, input_size):
        super(AQPredictor, self).__init__()
        self.model = AQPredictionModel(input_size)
        self.criterion = nn.MSELoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = self.criterion(output, labels.unsqueeze(dim=1)) if labels is not None else 0
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.0001)


def get_model(model_dir: str, version: int, path: str):
    model = AQPredictor.load_from_checkpoint(f"out/lightning_logs/{model_dir}/version_{version}/checkpoints/{path}",
                                             input_size=INPUT_SIZE)
    model.freeze()
    return model


def validate(model, X, y):
    sequences = create_sequences(X, y, SEQUENCE_LENGTH, test=True)
    dataset = AQDataset(sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predictions = []
    labels = []

    for item in dataset:
        sequence = item["sequence"]
        label = item["label"]

        sequence = sequence.unsqueeze(dim=0).to(device)

        _, output = model(sequence)
        predictions.append(int(np.expm1(output.item())))
        labels.append(int(np.expm1(label)))

    print(type(predictions[0]), type(labels[0]))

    rmse = np.sqrt(mean_squared_error(predictions, labels))

    return predictions, labels, rmse


def train_model(data_module):
    model = AQPredictor(INPUT_SIZE)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoint",
                                          filename="best-checkpoint",
                                          save_top_k=1,
                                          verbose=True,
                                          monitor="val_loss",
                                          mode="min")

    logger = TensorBoardLogger("./out/lightning_logs", name="pm2.5-change-fill")

    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            patience=3)

    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=checkpoint_callback,
                         callbacks=[early_stopping_callback],
                         max_epochs=NUM_EPOCHS,
                         gpus=1,
                         progress_bar_refresh_rate=30)

    for item in data_module.train_dataloader():
        print(item["sequence"].shape)
        print(item["label"].shape)
        break

    trainer.fit(model, data_module)

    return trainer, model
