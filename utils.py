import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

from data_loader import load_fits_data, load_fits_file, remove_underrepresented_classes
from dataset import get_time_series_dataset
from model import get_tft_model

def calculate_optimal_lengths(df, quantile=0.95):
    lengths = df.groupby("group_id")["time_idx"].max() + 1
    optimal_length = int(lengths.quantile(quantile))
    return optimal_length // 2, optimal_length // 2

def free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def findOptimumLr(file_path):
    pl.seed_everything(42)

    print("Lade Daten...")
    df = load_fits_file(file_path)

    print("Berechne optimale Längen für Encoder und Decoder...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df, quantile=0.95)
    print(f"Optimale max_encoder_length: {max_encoder_length}, max_prediction_length: {max_prediction_length}")
    min_encoder_length, min_prediction_length = calculate_optimal_lengths(df,quantile=0.05)
    print(f"Optimale min_encoder_length: {min_encoder_length}, min_prediction_length: {min_prediction_length}")

    remove_underrepresented_classes(df)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sim_type_index"])

    training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length, min_encoder_length, min_prediction_length)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=16, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=32, num_workers=16, persistent_workers=True) 

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3, 
        verbose=False,
        mode="min"
        )

    trainer = pl.Trainer(
        accelerator='cuda',
        devices=1,
        callbacks=[early_stop_callback],
        enable_progress_bar=True,
    )

    tft = get_tft_model(training)
    
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
        num_training=200
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    
def get_shortest_series(df, n=10):
    lengths = df.groupby("group_id")["time_idx"].max() + 1

    total_series = len(lengths)
    print(f"Gesamtanzahl der Datensätze im DataFrame: {total_series}")

    num_shorter_than_20 = (lengths < 20).sum()
    print(f"Anzahl der Datensätze, die kürzer als 20 Zeitpunkte sind: {num_shorter_than_20}")

    shortest_lengths = lengths.nsmallest(n)
    print("Längen der kürzesten Zeitreihen:\n", shortest_lengths)
    
    shortest_series = []

    for group_id in shortest_lengths.index:
        series = df[df["group_id"] == group_id]
        shortest_series.append(series)
        print(f"Zeitreihe für group_id {group_id}:\n", series)

    return shortest_series    

class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        free_memory()
