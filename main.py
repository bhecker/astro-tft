from train_model import train_model

if __name__ == "__main__":
    directory_path = '/Users/Benji/Documents/FOM Hochschule/Semester 7/Bachelorarbeit/fits'
    file_prefix = 'ELASTICC2_TRAIN_02_NONIaMODEL0-'
    
    tft_model, accuracy = train_model(directory_path, file_prefix)
