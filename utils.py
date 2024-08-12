from collections import defaultdict
import gc
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, Callback
from astropy.io import fits

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

    train_dataloader = training.to_dataloader(train=True, batch_size=256, num_workers=28, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=256*10, num_workers=28, persistent_workers=True) 

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3, 
        verbose=False,
        mode="min"
        )

    trainer = pl.Trainer(        
        accelerator="cuda",
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
    fig.savefig('lr_find_plot.png')
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

    train_dataloader = training.to_dataloader(train=True, batch_size=256, num_workers=28, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=256*10, num_workers=28, persistent_workers=True) 

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3, 
        verbose=False,
        mode="min"
        )

    trainer = pl.Trainer(        
        accelerator="cuda",
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
    fig.savefig('lr_find_plot.png')
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

def split_lightcurves():
    group_size = 1000
    with fits.open('test-lightcurves-cleaned.fits') as hdul:
        data = hdul[1].data
        columns = hdul[1].columns

    # Get unique group_ids
    group_ids = np.unique(data['group_id'])

    # Split the group_ids into chunks of specified size
    for i in range(0, len(group_ids), group_size):
        chunk_group_ids = group_ids[i:i + group_size]

        # Filter data to only include rows with group_ids in the current chunk
        chunk_data = data[np.isin(data['group_id'], chunk_group_ids)]

        # Create new FITS file
        hdu = fits.BinTableHDU.from_columns(columns, nrows=len(chunk_data))

        for colname in columns.names:
            hdu.data[colname] = chunk_data[colname]

        # Save the chunk to a new FITS file
        output_file = os.path.join('lightcurves', f"lightcurves_chunk_{i // group_size + 1}.fits")
        hdu.writeto(output_file, overwrite=True)
        print(f"Saved {len(chunk_group_ids)} group_ids to {output_file}")

def split_fits_file(file_path, output_dir):
    # Erstelle das Ausgabeverzeichnis, falls es noch nicht existiert
    os.makedirs(output_dir, exist_ok=True)

    # FITS-Datei laden
    with fits.open(file_path) as hdul:
        data = hdul[1].data

    # Gruppiere die Daten nach 'group_id'
    group_dict = defaultdict(list)
    for row in data:
        group_dict[row['group_id']].append(row)

    # Sortiere die Gruppen nach der Anzahl der Mitglieder
    sorted_groups = sorted(group_dict.items(), key=lambda x: len(x[1]), reverse=True)

    # Teile die Gruppen in zwei Hälften
    half_size = len(data) // 2
    current_size = 0
    first_half = []
    second_half = []
    for group_id, rows in sorted_groups:
        if current_size + len(rows) <= half_size:
            first_half.extend(rows)
            current_size += len(rows)
        else:
            second_half.extend(rows)

    # Erste Hälfte speichern
    save_fits(first_half, os.path.join(output_dir, 'half_1_' + os.path.basename(file_path)))

    # Zweite Hälfte speichern
    save_fits(second_half, os.path.join(output_dir, 'half_2_' + os.path.basename(file_path)))

def save_fits(data, output_file):
    # Konvertiere die Daten zurück in ein FITS-kompatibles Format
    col1 = fits.Column(name='group_id', format='A20', array=np.array([row['group_id'] for row in data]))
    col2 = fits.Column(name='time_idx', format='K', array=np.array([row['time_idx'] for row in data]))
    col3 = fits.Column(name='fluxcal', format='1E', array=np.array([row['fluxcal'] for row in data]))
    col4 = fits.Column(name='fluxcalerr', format='1E', array=np.array([row['fluxcalerr'] for row in data]))
    col5 = fits.Column(name='mjd', format='1D', array=np.array([row['mjd'] for row in data]))
    col6 = fits.Column(name='band', format='2A', array=np.array([row['band'] for row in data]))
    col7 = fits.Column(name='redshift', format='1E', array=np.array([row['redshift'] for row in data]))
    col8 = fits.Column(name='sim_type_index', format='K', array=np.array([row['sim_type_index'] for row in data]))

    # Erstelle die neue FITS-Tabelle
    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7, col8])

    # Speichere die FITS-Datei
    hdu.writeto(output_file, overwrite=True)

    print(f"Neue FITS-Datei erstellt: {output_file}")

def split_all_fits_files(directory_path, output_dir):
    fits_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.fits')]

    for file_path in fits_files:
        split_fits_file(file_path, output_dir)

def combine_files(output_dir, dataloader_id):
    # Set the path for the specific dataloader
    dataloader_dir = os.path.join(output_dir, str(dataloader_id))
    
    # Initialize lists to hold the combined data
    combined_class_predictions = []
    combined_true_labels = []
    combined_probabilities_avg = []

    # Sort the files for each type
    class_prediction_files = sorted([f for f in os.listdir(dataloader_dir) if f.startswith('class_predictions_batch_')],
                                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
    true_label_files = sorted([f for f in os.listdir(dataloader_dir) if f.startswith('true_labels_batch_')],
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
    probabilities_avg_files = sorted([f for f in os.listdir(dataloader_dir) if f.startswith('probabilities_avg_')],
                                     key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"Found {len(class_prediction_files)} class prediction files.")
    print(f"Found {len(true_label_files)} true label files.")
    print(f"Found {len(probabilities_avg_files)} probabilities avg files.")

    # Check if any files were found before attempting to concatenate
    if not class_prediction_files or not true_label_files or not probabilities_avg_files:
        print(f"No files found for dataloader {dataloader_id}. Skipping.")
        return

    # Combine each sorted type of file
    for class_file, label_file, prob_file in zip(class_prediction_files, true_label_files, probabilities_avg_files):
        # Load the class predictions, true labels, and probabilities for the current batch
        class_predictions = np.load(os.path.join(dataloader_dir, class_file))
        true_labels = np.load(os.path.join(dataloader_dir, label_file))
        probabilities_avg = np.load(os.path.join(dataloader_dir, prob_file))

        # Append to the combined lists
        combined_class_predictions.append(class_predictions)
        combined_true_labels.append(true_labels)
        combined_probabilities_avg.append(probabilities_avg)

    # Concatenate all batches to create the final arrays
    combined_class_predictions = np.concatenate(combined_class_predictions)
    combined_true_labels = np.concatenate(combined_true_labels)
    combined_probabilities_avg = np.concatenate(combined_probabilities_avg)

    # Save the combined arrays to new files
    np.save(os.path.join(output_dir, f'combined_class_predictions_dataloader_{dataloader_id}.npy'), combined_class_predictions)
    np.save(os.path.join(output_dir, f'combined_true_labels_dataloader_{dataloader_id}.npy'), combined_true_labels)
    np.save(os.path.join(output_dir, f'combined_probabilities_avg_dataloader_{dataloader_id}.npy'), combined_probabilities_avg)
    
    print(f"Combined files saved to {output_dir} for dataloader id {dataloader_id}")

def combine_all_dataloaders(output_dir):
    # Get all dataloader directories
    dataloader_dirs = [d for d in os.listdir(output_dir) if d.isdigit()]

    combined_class_predictions = []
    combined_true_labels = []
    combined_probabilities_avg = []

    for dataloader_dir in sorted(dataloader_dirs, key=int):
        dataloader_id = int(dataloader_dir)
        if int(dataloader_id) == 194:
            continue
        
        #combine_files(output_dir, dataloader_id)

        # Load the combined files for this dataloader
        class_predictions = np.load(os.path.join(output_dir, f'combined_class_predictions_dataloader_{dataloader_id}.npy'))
        true_labels = np.load(os.path.join(output_dir, f'combined_true_labels_dataloader_{dataloader_id}.npy'))
        probabilities_avg = np.load(os.path.join(output_dir, f'combined_probabilities_avg_dataloader_{dataloader_id}.npy'))

        # Append to the final combined lists
        combined_class_predictions.append(class_predictions)
        combined_true_labels.append(true_labels)
        combined_probabilities_avg.append(probabilities_avg)

    # Concatenate all dataloader arrays to create the final arrays
    final_class_predictions = np.concatenate(combined_class_predictions)
    final_true_labels = np.concatenate(combined_true_labels)
    final_probabilities_avg = np.concatenate(combined_probabilities_avg)

    # Save the final combined arrays to new files
    np.save(os.path.join(output_dir, 'final_combined_class_predictions.npy'), final_class_predictions)
    np.save(os.path.join(output_dir, 'final_combined_true_labels.npy'), final_true_labels)
    np.save(os.path.join(output_dir, 'final_combined_probabilities_avg.npy'), final_probabilities_avg)
    
    print(f"Final combined files saved to {output_dir}")


class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        free_memory()
