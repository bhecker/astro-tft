from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy

def get_tft_model(training, learning_rate=0.01, hidden_size=16, attention_head_size=1, dropout=0.1, hidden_continuous_size=8):
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=len(training.target_normalizer.classes_),
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
