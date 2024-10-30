import time

from data_loader import read_fits_to_new_fits
from utils import combine_all_dataloaders, findOptimumLr, split_lightcurves
from train_model import process_all_fits_files, train_model
from metrics import calculate_metrics

#torch.set_float32_matmul_precision('medium')
#os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.80'
#os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.85'

if __name__ == "__main__":
    directory_path = 'lightcurve-fits'
    test_directory_path = 'test-lightcurves'
    file_prefix = 'ELASTICC2_TRAIN_02_NONIaMODEL0-'
    checkpoint_path = 'best-checkpoint.ckpt'
    file_path = 'lightcurves.fits'
    file_path_test = 'test-lightcurves.fits'
    predictions_path = 'predictions-test'

    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    read_fits_to_new_fits(directory_path, file_prefix, file_path)
    read_fits_to_new_fits(directory_path, file_prefix, file_path_test)

    optimum_lr = findOptimumLr(file_path)    

    tft_model = train_model(file_path)
    
    split_lightcurves(file_path_test, test_directory_path, test_directory_path)
    process_all_fits_files(test_directory_path, checkpoint_path)
    combine_all_dataloaders(predictions_path)
    calculate_metrics()

    end_time = time.time()
    print(f"Training beendet um: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Dauer des Trainings: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")