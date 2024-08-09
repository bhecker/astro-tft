import subprocess

def install_packages_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        # Die erste Zeile ist die Kopfzeile und sollte ignoriert werden
        if line.startswith('Package') or line.startswith('------'):
            continue
        
        # Jedes Paket ist durch Leerzeichen getrennt; der erste Teil ist der Name, der zweite die Version
        parts = line.split()
        package_name = parts[0]
        package_version = parts[1]
        
        # Installiere das Paket mit der angegebenen Version
        package_with_version = f"{package_name}=={package_version}"
        subprocess.run(['pip', 'install', package_with_version], check=True)

if __name__ == "__main__":
    # Ersetze 'packages.txt' durch den Pfad zu deiner Datei
    install_packages_from_txt('piplistLambda.txt')