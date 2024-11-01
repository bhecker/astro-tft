import os
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
import torch

def get_tft_model(training, learning_rate=0.3, lstm_layers=2, hidden_size=128, attention_head_size=4, dropout=0.2, hidden_continuous_size=8):
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        lstm_layers=lstm_layers,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=len(training.target_normalizer.classes_),
        loss=CrossEntropy(),
        log_interval=30,
        reduce_on_plateau_patience=2,
        optimizer="Ranger"
    )

    tft.save_hyperparameters(ignore=['loss', 'logging_metrics'])

    return tft

def get_best_tft_model(best_model_path):
    return TemporalFusionTransformer.load_from_checkpoint(best_model_path, map_location=torch.device('mps'))