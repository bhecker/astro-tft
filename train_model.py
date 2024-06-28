import lightning.pytorch as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from lightning.pytorch.callbacks import EarlyStopping

from data_loader import load_fits_data
from dataset import get_time_series_dataset
from model import get_tft_model

def train_model(directory_path, file_prefix, max_encoder_length=24, max_prediction_length=6, batch_size=64, max_epochs=30):
    print("Lade Daten...")
    df = load_fits_data(directory_path, file_prefix)
    
    print("DataFrame Vorschau:\n", df.head())
    print("Anzahl der Zeilen im DataFrame:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame ist leer. Überprüfen Sie die Datenlade- und Verarbeitungsfunktionen.")
    
    print("Teile Daten in Trainings- und Validierungssets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["group_id"])
    
    training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    
    tft = get_tft_model(training)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")

    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    tft = tft.to(device)

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator=device,
        devices=1,
        callbacks=[early_stop_callback],
        enable_progress_bar=True
    )
    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    train_predictions = tft.predict(train_dataloader, mode="raw").to(device)
    train_y_true = torch.cat([y["decoder_target"].to(device).float() for x, y in train_dataloader], dim=0).cpu().numpy()
    train_y_pred = torch.cat([pred.argmax(dim=1).to(device).float() for pred in train_predictions], dim=0).cpu().numpy()
    train_accuracy = accuracy_score(train_y_true, train_y_pred)

    val_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
    val_y_true = torch.cat([y["decoder_target"].to(device).float() for x, y in val_dataloader], dim=0).cpu().numpy()
    val_y_pred = torch.cat([pred.argmax(dim=1).to(device).float() for pred in val_predictions], dim=0).cpu().numpy()
    val_accuracy = accuracy_score(val_y_true, val_y_pred)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print()
        
    return tft, val_accuracy
