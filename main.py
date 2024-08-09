import os
from astropy.io import fits
from pytorch_forecasting import TemporalFusionTransformer
import torch

from data_loader import get_fits_column_names, load_fits_data, read_fits_to_new_fits, read_fits_to_new_test_fits
#os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.80'
#os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.85'

import time
from train_model import find_optimal_hyperparameters_from_saved_model, predict_from_saved_model, train_model
from utils import findOptimumLr, get_shortest_series
import pandas as pd
from astropy.io import fits

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    directory_path = 'training_data/test_fits'
    file_prefix = 'ELASTICC2_TRAIN_02_NONIaMODEL0-'
    checkpoint_path = 'best-checkpoint.ckpt'
    file_path = 'test-lightcurves-cleaned.fits'

    start_time = time.time()
    print(f"Training gestartet um: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    #read_fits_to_new_fits(directory_path, file_prefix, "test-lightcurves.fits")
    #read_fits_to_new_test_fits(directory_path, file_prefix, "test-cleaned-lightcurves.fits")

    #optimum_lr = findOptimumLr(file_path)    

    #tft_model = train_model(file_path)
        
    tft_model = predict_from_saved_model(file_path, checkpoint_path)

    #find_optimal_hyperparameters_from_saved_model(directory_path, file_prefix, checkpoint_path)

    end_time = time.time()
    print(f"Training beendet um: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Dauer des Trainings: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")