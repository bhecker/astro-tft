import numpy as np
import pandas as pd
from astropy.io import fits
import glob
import os

def load_fits_file(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        data = data.byteswap().newbyteorder()
        df = pd.DataFrame(np.array(data))

    return df

def load_fits_data(directory_path, file_prefix):
    head_files = sorted(glob.glob(os.path.join(directory_path, f"**/{file_prefix}*HEAD.FITS")))
    phot_files = sorted(glob.glob(os.path.join(directory_path, f"**/{file_prefix}*PHOT.FITS")))

    data = []

    print(f"Found HEAD-files: {len(head_files)}")
    print(f"Found PHOT-files: {len(phot_files)}")

    for head_file, phot_file in zip(head_files, phot_files):
        print(f"Processing HEAD-file: {head_file}")
        print(f"Processing PHOT-file: {phot_file}")

        with fits.open(head_file) as head_hdu:
            head_data = head_hdu[1].data

        with fits.open(phot_file) as phot_hdu:
            phot_data = phot_hdu[1].data

            for head_row in head_data:
                snid = head_row['SNID']
                ptrobs_min = head_row['PTROBS_MIN']
                ptrobs_max = head_row['PTROBS_MAX']
                sim_type_index = head_row['SIM_TYPE_INDEX']

                light_curve = phot_data[ptrobs_min:ptrobs_max]
                fluxcal_values = light_curve['FLUXCAL']

                for time_idx, fluxcal in enumerate(fluxcal_values):
                    data.append([snid, time_idx, fluxcal, sim_type_index])

        print(f"Completed: {head_file} and {phot_file}")

    df = pd.DataFrame(data, columns=["group_id", "time_idx", "fluxcal", "sim_type_index"])
    
    print(f"Created DataFrame with {len(df)} lines")
    return df

def read_fits_to_new_fits(directory_path, file_prefix, output_file):
    head_files = sorted(glob.glob(os.path.join(directory_path, f"**/{file_prefix}*HEAD.FITS"), recursive=True))
    phot_files = sorted(glob.glob(os.path.join(directory_path, f"**/{file_prefix}*PHOT.FITS"), recursive=True))

    def filter_files(file_list):
        filtered_files = []
        for f in file_list:
            number = int(os.path.basename(f).split('-')[-1].split('_')[0])
            if 1 <= number <= 5:
                filtered_files.append(f)
        return filtered_files

    head_files = filter_files(head_files)
    phot_files = filter_files(phot_files)

    data = []

    print(f"Found HEAD-files: {len(head_files)}")
    print(f"Found PHOT-files: {len(phot_files)}")

    for head_file, phot_file in zip(head_files, phot_files):
        print(f"Processing HEAD-file: {head_file}")
        print(f"Processing PHOT-file: {phot_file}")

        with fits.open(head_file) as head_hdu:
            head_data = head_hdu[1].data

        with fits.open(phot_file) as phot_hdu:
            phot_data = phot_hdu[1].data

            for head_row in head_data:
                snid = head_row['SNID']
                ptrobs_min = head_row['PTROBS_MIN']
                ptrobs_max = head_row['PTROBS_MAX']
                sim_type_index = head_row['SIM_TYPE_INDEX']
                redshift = head_row['REDSHIFT_HELIO']                

                light_curve = phot_data[ptrobs_min:ptrobs_max]
                mjd = light_curve['MJD']
                fluxcal_values = light_curve['FLUXCAL']
                fluxcal_err = light_curve['FLUXCALERR']
                band = light_curve['BAND']

                for time_idx, (fluxcal, fluxcal_err_val, mjd_val, band_val) in enumerate(zip(fluxcal_values, fluxcal_err, mjd, band)):
                    data.append((snid, time_idx, fluxcal, fluxcal_err_val, mjd_val, band_val, redshift, sim_type_index))

        print(f"Completed: {head_file} and {phot_file}")

    col1 = fits.Column(name='group_id', format='A20', array=np.array([row[0] for row in data]))
    col2 = fits.Column(name='time_idx', format='K', array=np.array([row[1] for row in data]))
    col3 = fits.Column(name='fluxcal', format='1E', array=np.array([row[2] for row in data]))
    col4 = fits.Column(name='fluxcalerr', format='1E', array=np.array([row[3] for row in data]))
    col5 = fits.Column(name='mjd', format='1D', array=np.array([row[4] for row in data]))
    col6 = fits.Column(name='band', format='2A', array=np.array([row[5] for row in data]))
    col7 = fits.Column(name='redshift', format='1E', array=np.array([row[6] for row in data]))
    col8 = fits.Column(name='sim_type_index', format='K', array=np.array([row[7] for row in data]))

    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7, col8])

    hdu.writeto(output_file, overwrite=True)
    
    print(f"Create new fits-File: {output_file}")

def get_fits_column_names(directory_path, file_prefix):
    head_file = sorted(glob.glob(os.path.join(directory_path, f"{file_prefix}*HEAD.FITS"), recursive=True))[0]
    phot_file = sorted(glob.glob(os.path.join(directory_path, f"{file_prefix}*PHOT.FITS"), recursive=True))[0]

    column_info = {}

    with fits.open(head_file) as hdu_list:
        columns = hdu_list[1].columns
        column_info['HEAD'] = [(col.name, col.format) for col in columns]

    with fits.open(phot_file) as hdu_list:
        columns = hdu_list[1].columns
        column_info['PHOT'] = [(col.name, col.format) for col in columns]

    print("HEAD fits column names and formats:")
    for name, fmt in column_info['HEAD']:
        print(f"Name: {name}, Format: {fmt}")

    print("\nPHOT fits column names and formats:")
    for name, fmt in column_info['PHOT']:
        print(f"Name: {name}, Format: {fmt}")

def check_inhomogeneous_shape(data, target_shape):
    for i, row in enumerate(data):
        value = row[3]
        try:
            np_array = np.array(value)
            if np_array.shape == target_shape:
                print(f"Row {i} has target shape {target_shape}: {value}")
        except Exception as e:
            print(f"Error processing row {i}: {e}")

def remove_underrepresented_classes(df):
    class_counts = df['sim_type_index'].value_counts()
    print("Classes before adjustment:")
    print(class_counts)

    few_members_classes = class_counts[class_counts < 2].index.tolist()
    print(f"Classes with less than 2 occurrences: {few_members_classes}")

    df = df[~df['sim_type_index'].isin(few_members_classes)]

    class_counts_after = df['sim_type_index'].value_counts()
    print("Classes after adjustment:")
    print(class_counts_after)

    return df