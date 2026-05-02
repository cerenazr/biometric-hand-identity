"""
01_download_data.py
-------------------
11k Hands Dataset'i indirir.

SEÇENEK A (Otomatik) - kagglehub:
  1. kaggle.com → Account → Settings → "Create New API Token" → kaggle.json indir
  2. C:/Users/<kullanici>/.kaggle/kaggle.json konumuna koy
  3. Bu scripti çalıştır

SEÇENEK B (Manuel):
  1. https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset adresine git
  2. Zip'i tarayıcıdan indir
  3. data/11k_hands/ klasorune cikar (HandInfo.csv bu klasorde olmali)
"""

import os
import shutil
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "11k_hands")
DATASET_SLUG = "shyambhu/hands-and-palm-images-dataset"


def check_already_downloaded():
    csv_path = os.path.join(DATA_DIR, "HandInfo.csv")
    if os.path.exists(csv_path):
        img_count = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"Dataset zaten mevcut: {img_count} görüntü, HandInfo.csv bulundu.")
        return True
    return False


def download_with_kagglehub():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub kurulu değil. Kurmak için: pip install kagglehub")
        return False

    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        print("\nKaggle API kimlik bilgisi bulunamadı!")
        print(f"Beklenen konum: {kaggle_json}")
        print("\nAdımlar:")
        print("  1. https://www.kaggle.com adresine git, hesabına giriş yap")
        print("  2. Sağ üst köşe → profil resmi → Settings")
        print("  3. 'API' bölümüne in → 'Create New Token' butonuna tıkla")
        print("  4. İnen kaggle.json dosyasını şuraya koy:")
        print(f"     {kaggle_json}")
        print("\nveya SEÇENEK B: dataset'i tarayıcıdan manuel indir.")
        return False

    print("Kaggle API ile indiriliyor...")
    try:
        cache_path = kagglehub.dataset_download(DATASET_SLUG)
        print(f"Kaggle cache konumu: {cache_path}")
        _copy_to_data_dir(cache_path)
        return True
    except Exception as e:
        print(f"İndirme hatası: {e}")
        return False


def _copy_to_data_dir(src_root):
    print(f"Dosyalar {DATA_DIR} klasörüne kopyalanıyor...")
    copied = 0
    for root, _, files in os.walk(src_root):
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(DATA_DIR, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
    print(f"{copied} dosya kopyalandı.")


def verify():
    csv_path = os.path.join(DATA_DIR, "HandInfo.csv")
    if not os.path.exists(csv_path):
        print("HATA: HandInfo.csv bulunamadı. Manuel indirme talimatlarına bak.")
        return False

    img_count = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"\nDoğrulama tamamlandı:")
    print(f"  HandInfo.csv : OK")
    print(f"  Görüntü sayısı: {img_count}")

    if img_count < 100:
        print("UYARI: Görüntü sayısı çok az, indirme tam olmamış olabilir.")
        return False

    print("\n01_download_data.py tamamlandı. Sıradaki adım:")
    print("  python 02_extract_features.py")
    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if check_already_downloaded():
        verify()
        return

    success = download_with_kagglehub()

    if not success:
        print("\n--- MANUEL İNDİRME TALİMATLARI ---")
        print("1. https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset adresine git")
        print("2. 'Download' butonuna tıkla (giriş gerekli)")
        print("3. İnen zip dosyasını aç")
        print(f"4. İçindeki tüm dosyaları şuraya çıkar:")
        print(f"   {DATA_DIR}")
        print("5. HandInfo.csv bu klasörde görünmeli")
        print("6. Tekrar çalıştır: python 01_download_data.py")
        sys.exit(1)

    verify()


if __name__ == "__main__":
    main()
