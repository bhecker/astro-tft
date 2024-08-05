import os
import pickle
import lightning.pytorch as pl
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import seaborn as sns
import torch.nn.functional as F
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


from data_loader import load_fits_data
from dataset import get_time_series_dataset
from model import get_best_tft_model, get_tft_model
from utils import MemoryCleanupCallback, calculate_optimal_lengths, free_memory

def train_model(directory_path, file_prefix, batch_size=32, max_epochs=10):
    pl.seed_everything(42)

    free_memory()

    print("Lade Daten...")
    df = load_fits_data(directory_path, file_prefix)
    
    print("DataFrame Vorschau:\n", df.head())
    print("Anzahl der Zeilen im DataFrame:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame ist leer. Überprüfen Sie die Datenlade- und Verarbeitungsfunktionen.")
    
    print("Berechne optimale Längen für Encoder und Decoder...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df)
    print(f"Optimale max_encoder_length: {max_encoder_length}, max_prediction_length: {max_prediction_length}")


    print("Teile Daten in Trainings- und Validierungssets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["group_id"])
    
    training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=False)
    
    tft = get_tft_model(training, learning_rate=0.03)

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

    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator=device,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback, memory_cleanup_callback],
        enable_progress_bar=True,
        logger=logger
    )
    
    free_memory()
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    free_memory()

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = get_best_tft_model(best_model_path)

    trainer_kwargs = {
        'accelerator': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'devices': 1,
        'enable_progress_bar': True 
    }

    with torch.no_grad():
        train_predictions = tft.predict(train_dataloader, mode="raw", trainer_kwargs=trainer_kwargs, return_x=True)
        train_y_true = torch.cat([y["decoder_target"] for x, y in train_dataloader], dim=0).numpy()
        train_y_pred = torch.cat([pred.argmax(dim=1) for pred in train_predictions], dim=0).numpy()
        train_accuracy = accuracy_score(train_y_true, train_y_pred)

        val_predictions, x = tft.predict(tft, val_dataloader, mode="raw",trainer_kwargs=trainer_kwargs, return_x=True)
        val_y_true = torch.cat([y["decoder_target"] for x, y in val_dataloader], dim=0).numpy()
        val_y_pred = torch.cat([pred.argmax(dim=1) for pred in val_predictions], dim=0).numpy()
        val_accuracy = accuracy_score(val_y_true, val_y_pred)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print()
        
    return tft, val_accuracy

def predict_from_saved_model(directory_path, file_prefix, best_model_path, batch_size=128):
    pl.seed_everything(42)

    free_memory()

    print("Lade Daten...")
    df = load_fits_data(directory_path, file_prefix)
    
    print("DataFrame Vorschau:\n", df.head())
    print("Anzahl der Zeilen im DataFrame:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame ist leer. Überprüfen Sie die Datenlade- und Verarbeitungsfunktionen.")
    
    print("Berechne optimale Längen für Encoder und Decoder...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df, quantile=0.95)
    print(f"Optimale max_encoder_length: {max_encoder_length}, max_prediction_length: {max_prediction_length}")
    min_encoder_length, min_prediction_length = calculate_optimal_lengths(df,quantile=0.05)
    print(f"Optimale min_encoder_length: {min_encoder_length}, min_prediction_length: {min_prediction_length}")

    print("Teile Daten in Trainings- und Validierungssets...")
    #train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["group_id"])
    test = get_time_series_dataset(df, max_encoder_length, max_prediction_length, min_encoder_length, min_prediction_length)
    #training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length, min_encoder_length, min_prediction_length)
    #validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
#
    #train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4, persistent_workers=True)
    #val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=False)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=False)
    tft = get_best_tft_model(best_model_path)

    trainer_kwargs = {
        'accelerator': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'devices': 1,
        'enable_progress_bar': True,
    }

    # print("TRAINING")

    # with torch.inference_mode():
    #     #train_predictions = tft.predict(train_dataloader, mode="raw", trainer_kwargs=trainer_kwargs, return_x=True)
    #     #train_y_true = torch.cat([y["decoder_target"] for x, y in train_dataloader], dim=0).numpy()
    #     #train_y_pred = torch.cat([pred.argmax(dim=1) for pred in train_predictions], dim=0).numpy()
    #     #train_accuracy = accuracy_score(train_y_true, train_y_pred)

    #     """ predictions, inputs = tft.predict(val_dataloader, 
    #                 mode="prediction",
    #                 trainer_kwargs=trainer_kwargs, 
    #                 #write_interval='batch', 
    #                 #output_dir='predictions', 
    #                 fast_dev_run=True,
    #                 return_x=True) """
    #     tft.predict(train_dataloader, 
    #                 mode="raw",
    #                 write_interval='batch',
    #                 output_dir='predictions',
    #                 trainer_kwargs=trainer_kwargs,
    #                 return_x=True)
    
    # free_memory()

    # train_class_predictions = []
    # train_true_labels = []

    # length_dataloader = len(train_dataloader)

    # for batch_idx in range(length_dataloader):
    #     train_class_predictions.append(np.load(f'predictions/train_dataloader_0/class_predictions_batch_{batch_idx}.npy'))
    #     train_true_labels.append(np.load(f'predictions/train_dataloader_0/true_labels_batch_{batch_idx}.npy'))

    # train_class_predictions = np.concatenate(train_class_predictions)
    # train_true_labels = np.concatenate(train_true_labels)

    # print("Train class pred shape:", train_class_predictions.shape)
    # print("Train true labels shape:", train_true_labels.shape)

    # train_accuracy = accuracy_score(train_true_labels, train_class_predictions)
    # print(f"Train Accuracy: {train_accuracy:.4f}")

    # original_class_names = ['Cepheids', 'Dwarf novae', 'Lenses', 'SNIIa']

    # print(classification_report(train_true_labels, train_class_predictions, target_names=original_class_names, zero_division=0))

    # conf_matrix = confusion_matrix(train_true_labels, train_class_predictions, labels=original_class_names)

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix Training')
    # plt.show()

    # del train_class_predictions
    # del train_true_labels

    # free_memory()

    print("VALIDATION")
    tft.eval()
    #with torch.inference_mode():
        #tft.predict(test_dataloader, 
        #            mode="raw",
        #            trainer_kwargs=trainer_kwargs, 
        #            write_interval='batch',
        #            output_dir='predictions',
        #            return_x=True)
    free_memory()

    val_class_predictions = []
    val_true_labels = []

    length_dataloader = len(test_dataloader)

    for batch_idx in range(6802):
        val_class_predictions.append(np.load(f'predictions/test_dataloader_0/class_predictions_batch_{batch_idx}.npy'))
        val_true_labels.append(np.load(f'predictions/test_dataloader_0/true_labels_batch_{batch_idx}.npy'))

    # Concatenate all batches
    val_class_predictions = np.concatenate(val_class_predictions)
    val_true_labels = np.concatenate(val_true_labels)
            
    print("Val class pred shape:", val_class_predictions.shape)
    print("Val true labels shape:", val_true_labels.shape)

    val_accuracy = accuracy_score(val_true_labels, val_class_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    original_class_names = ['Cepheids', 'Dwarf novae', 'Lenses', 'SNIa']

    print(classification_report(val_true_labels, val_class_predictions, target_names=original_class_names, zero_division=0))

    conf_matrix = confusion_matrix(val_true_labels, val_class_predictions, labels=original_class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Validation')
    plt.show()

    del val_class_predictions
    del val_true_labels

    free_memory()
#Felder von predictions.output
#_fields: ('prediction', 'encoder_attention', 'decoder_attention', 'static_variables', 'encoder_variables', 'decoder_variables', 'decoder_lengths', #'encoder_lengths')

#Keys in predictions.x: dict_keys(['encoder_cat', 'encoder_cont', 'encoder_target', 'encoder_lengths', 'decoder_cat', 'decoder_cont', #'decoder_target', 'decoder_lengths', 'decoder_time_idx', 'groups', 'target_scale'])

    return tft, val_accuracy

def find_optimal_hyperparameters_from_saved_model(directory_path, file_prefix, best_model_path, batch_size=32):
    pl.seed_everything(42)

    free_memory()

    print("Lade Daten...")
    df = load_fits_data(directory_path, file_prefix)
    print("unique werte", df['group_id'].nunique())
    print("DataFrame Vorschau:\n", df.head())
    print("Anzahl der Zeilen im DataFrame:", len(df))
    
    if df.empty:
        raise ValueError("DataFrame ist leer. Überprüfen Sie die Datenlade- und Verarbeitungsfunktionen.")
    
    print("Berechne optimale Längen für Encoder und Decoder...")
    max_encoder_length, max_prediction_length = calculate_optimal_lengths(df)
    min_encoder_length, min_prediction_length = calculate_optimal_lengths(df,quantile=0.05)


    print("Teile Daten in Trainings- und Validierungssets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["group_id"])
    
    training = get_time_series_dataset(train_df, max_encoder_length, max_prediction_length, min_encoder_length, min_prediction_length)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=False)

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
        learning_rate_range=(0.01, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30, accelerator='mps', devices=1, enable_progress_bar=True),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False
    )

    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    print(study.best_trial.params)