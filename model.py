import os
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
import torch

def get_tft_model(training, learning_rate=0.01, hidden_size=4, attention_head_size=1, dropout=0.1, hidden_continuous_size=4):
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=len(training.target_normalizer.classes_),
        loss=CrossEntropy(),
        log_interval=30,
        reduce_on_plateau_patience=4
    )

    tft.save_hyperparameters(ignore=['loss', 'logging_metrics'])

    return tft

def get_best_tft_model(best_model_path):
    return TemporalFusionTransformer.load_from_checkpoint(best_model_path)