import pandas as pd
from astropy.io import fits
import glob
import os

def load_fits_data(directory_path, file_prefix):
    head_files = sorted(glob.glob(os.path.join(directory_path, f"{file_prefix}*HEAD.FITS")))
    phot_files = sorted(glob.glob(os.path.join(directory_path, f"{file_prefix}*PHOT.FITS")))

    data = []

    print(f"Gefundene HEAD-Dateien: {len(head_files)}")
    print(f"Gefundene PHOT-Dateien: {len(phot_files)}")

    for head_file, phot_file in zip(head_files, phot_files):
        print(f"Verarbeite HEAD-Datei: {head_file}")
        print(f"Verarbeite PHOT-Datei: {phot_file}")

        with fits.open(head_file) as head_hdu:
            head_data = head_hdu[1].data

        with fits.open(phot_file) as phot_hdu:
            phot_data = phot_hdu[1].data

            for head_row in head_data:
                snid = head_row['SNID']
                ptrobs_min = head_row['PTROBS_MIN']
                ptrobs_max = head_row['PTROBS_MAX']
                sim_type_index = head_row['SIM_TYPE_INDEX']

                light_curve = phot_data[ptrobs_min:ptrobs_max + 1]
                fluxcal_values = light_curve['FLUXCAL']

                for time_idx, fluxcal in enumerate(fluxcal_values):
                    data.append([snid, time_idx, fluxcal, sim_type_index])

        print(f"Abgeschlossen: {head_file} und {phot_file}")

    df = pd.DataFrame(data, columns=["group_id", "time_idx", "fluxcal", "sim_type_index"])
    
    df['time_idx'] = df['time_idx'].astype('int32')
    df['fluxcal'] = df['fluxcal'].astype('float32')
    df['sim_type_index'] = df['sim_type_index'].astype('int32')

    print(f"Datentypen des DataFrames:\n{df.dtypes}")

    print(f"Erstelltes DataFrame mit {len(df)} Zeilen")
    return df