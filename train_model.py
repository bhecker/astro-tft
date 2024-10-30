import gc
import glob
import os
import pickle
import lightning.pytorch as pl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import seaborn as sns
import torch.nn.functional as F
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from astropy.io import fits

from data_loader import load_fits_file
from dataset import get_time_series_dataset
from model import get_best_tft_model, get_tft_model
from utils import MemoryCleanupCallback, calculate_optimal_lengths, free_memory

def train_model(file_path, batch_size=32, learning_rate=0.01, max_epochs=10, num_workers=0):
    pl.seed_everything(42)

    free_memory()

    print("Load data...")
    df = load_fits_file(file_path)
    
    print("DataFrame preview:\n", df.head())
    print("DataFrame line count:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame is empty. Check dataloading.")
    
    print("Calculating optimal encoder length...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df, 0.95), 1
    print(f"Optimal max_encoder_length: {max_encoder_length}")
    min_encoder_length, min_prediction_length = calculate_optimal_lengths(df, 0.05), 1
    print(f"Optimal min_encoder_length: {min_encoder_length}")

    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sim_type_index"])
    
    training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length, min_prediction_length, max_prediction_length)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=num_workers, persistent_workers=True, shuffle=False)
    
    tft = get_tft_model(training, learning_rate=learning_rate)

    logger = TensorBoardLogger("tb_logs", name="my_model")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3, 
        verbose=False,
        mode="min"
        )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    memory_cleanup_callback = MemoryCleanupCallback()

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator='cuda',
    	devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback, memory_cleanup_callback],
        enable_progress_bar=True,
        logger=logger
    )
    
    free_memory()
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    free_memory()
            
    return tft

def find_optimal_hyperparameters_from_saved_model(file_path, best_model_path, batch_size=128):
    pl.seed_everything(42)

    free_memory()

    print("Load data...")
    df = load_fits_file(file_path)
    
    print("DataFrame preview:\n", df.head())
    print("DataFrame line count:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame is empty. Check dataloading.")
    
    print("Calculating optimal encoder length...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df, 0.95), 1
    print(f"Optimal max_encoder_length: {max_encoder_length}")
    min_encoder_length, min_prediction_length = calculate_optimal_lengths(df, 0.05), 1
    print(f"Optimal min_encoder_length: {min_encoder_length}")    

    free_memory()

    print("Load data...")
    df = load_fits_file(file_path)
    
    print("DataFrame preview:\n", df.head())
    print("DataFrame line count:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame is empty. Check dataloading.")
    
    print("Calculating optimal encoder length...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df, 0.95), 1
    print(f"Optimal max_encoder_length: {max_encoder_length}")
    min_encoder_length, min_prediction_length = calculate_optimal_lengths(df, 0.05), 1
    print(f"Optimal min_encoder_length: {min_encoder_length}")

    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["group_id"])
    
    training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length, min_encoder_length, min_prediction_length)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=14, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=14, persistent_workers=True, shuffle=False)

    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path=best_model_path,
        n_trials=100,
        max_epochs=10,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(3, 4),
        hidden_continuous_size_range=(3, 4),
        attention_head_size_range=(1, 2),
        learning_rate_range=(0.01, 0.5),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30, accelerator='cuda', devices=1, enable_progress_bar=True),
        reduce_on_plateau_patience=3,
        use_learning_rate_finder=False
    )

    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    print(study.best_trial.params)

def process_all_fits_files(directory_path, checkpoint_path, batch_size=1028, max_encoder_length=200, min_encoder_length=1, min_pred_length=1, max_pred_length=1):
    pl.seed_everything(42)

    fits_files = sorted(glob.glob(os.path.join(directory_path, '*.fits')))

    tft = get_best_tft_model(checkpoint_path)
    
    trainer_kwargs = {
        'accelerator': 'cuda',
        'devices': 1,
        'enable_progress_bar': True,
    }
    
    i = 0
    for fits_file in fits_files:
        i = i + 1
        print("Predicting file", fits_file)
        print("for run No.", i)
        with fits.open(fits_file) as hdul:
            data = hdul[1].data
        
        df = pd.DataFrame(data)
        test = get_time_series_dataset(df, max_encoder_length, max_pred_length, min_encoder_length, min_pred_length)

        test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0, shuffle=False)
        
        predictions = tft.predict(test_dataloader, 
                                  mode="raw",
                                  trainer_kwargs=trainer_kwargs, 
                                  fast_dev_run=False,
                                  write_interval='batch',
                                  output_dir=f'predictions-test/{i}',
                                  return_x=True)

        gc.collect()
        torch.cuda.empty_cache()