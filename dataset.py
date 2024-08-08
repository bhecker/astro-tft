from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

def get_time_series_dataset(df, max_encoder_length, max_prediction_length, min_encoder_length, min_prediction_length):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="sim_type_index",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=min_encoder_length,
        min_prediction_length=min_prediction_length,
        time_varying_known_reals=["time_idx", "fluxcalerr", "mjd"],
        time_varying_unknown_reals=["fluxcal"],
        time_varying_unknown_categoricals=["band"],
        static_reals=["redshift"],
        target_normalizer=NaNLabelEncoder(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
