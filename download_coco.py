import os
import urllib.request
import zipfile
from tqdm import tqdm

# --- CONFIG ---
# Le gros dataset d'entrainement (18 Go, 118k images)
URL = "http://images.cocodataset.org/zips/train2017.zip"
DATA_ROOT = "./data/coco" # On met tout dans le dossier coco
ZIP_PATH = os.path.join(DATA_ROOT, "train2017.zip")
EXTRACT_PATH = DATA_ROOT # Ça créera un dossier 'train2017' dedans

def main():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    # 1. Téléchargement
    if not os.path.exists(ZIP_PATH):
        print(f"🌊 Téléchargement de COCO TRAIN 2017 (18 Go !)...")
        print("☕ Prends un café, ça va être long.")
        print(f"   URL: {URL}")
        
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
            def reporthook(blocknum, blocksize, totalsize):
                t.total = totalsize
                t.update(blocknum * blocksize - t.n)
            
            # On ajoute un timeout généreux pour les grosses connexions
            urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=reporthook)
        print("✅ Téléchargement terminé.")
    else:
        print("✅ Fichier ZIP déjà présent.")

    # 2. Extraction
    if not os.path.exists(os.path.join(DATA_ROOT, "train2017")):
        print(f"📦 Extraction (ça aussi c'est long)...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("✅ Extraction terminée.")
    else:
        print("✅ Dossier 'train2017' déjà extrait.")

    # Nettoyage zip pour gagner 18Go de place (Décommente si tu veux)
    # os.remove(ZIP_PATH)
    
    print(f"🚀 Tout est prêt. Tu as maintenant:")
    print(f"   - Train: 118 000 images")
    print(f"   - Val  :   5 000 images (si tu as gardé l'ancien téléchargement)")

if __name__ == "__main__":
    main()