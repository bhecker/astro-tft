import os
import gzip
import shutil
import re

pattern = re.compile(r"-0010.*\.gz$")

for root, dirs, files in os.walk('lightcurve-fits'):
    for file in files:
        if file.endswith(".gz") and pattern.search(file):
            # Erstelle den vollständigen Pfad zur .gz-Datei
            gz_file_path = os.path.join(root, file)
            # Erstelle den Pfad für die entpackte Datei (ohne .gz Endung)
            output_file_path = os.path.join(root, file[:-3])
            
            # Entpacken der .gz Datei
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Lösche die ursprüngliche .gz Datei
            os.remove(gz_file_path)
            print(f"Entpackt und gelöscht: {gz_file_path}")