import os
from PIL import Image
import random

def check_data_integrity():
    base_dir = r"C:\Users\yasin\Desktop\GeoHarvest_AI\EuroSAT_RGB"
    print("🔍 1. Klasör Yapısı Kontrolü")
    
    # Gerekli klasörlerin varlığını kontrol et
    required_dirs = [
        os.path.join(base_dir, "Data", "Train", "AnnualCrop"),
        os.path.join(base_dir, "Data", "Test", "HerbaceousVegetation"),
        os.path.join(base_dir, "processed_128", "train", "AnnualCrop"),
        os.path.join(base_dir, "processed_128", "Test", "HerbaceousVegetation")
    ]
    
    all_dirs_exist = all(os.path.exists(d) for d in required_dirs)
    print(f"✔ Tüm gerekli klasörler mevcut" if all_dirs_exist else "❌ Eksik klasörler bulundu")

    print("\n📊 2. Veri Dağılımı Kontrolü (80-20 Oranı)")
    
    # Ham veri oran kontrolü
    original_counts = {
        "train": {
            "AnnualCrop": len(os.listdir(os.path.join(base_dir, "Data", "Train", "AnnualCrop"))),
            "Herbaceous": len(os.listdir(os.path.join(base_dir, "Data", "Train", "HerbaceousVegetation")))
        },
        "test": {
            "AnnualCrop": len(os.listdir(os.path.join(base_dir, "Data", "Test", "AnnualCrop"))),
            "Herbaceous": len(os.listdir(os.path.join(base_dir, "Data", "Test", "HerbaceousVegetation")))
        }
    }
    
    total = original_counts["train"]["AnnualCrop"] + original_counts["test"]["AnnualCrop"]
    train_ratio = original_counts["train"]["AnnualCrop"]/total
    print(f"AnnualCrop Oranı: Train: {train_ratio:.2%} - Test: {1-train_ratio:.2%}")

    print("\n🖼 3. Ön İşleme Kontrolleri")
    
    # Boyut ve format kontrolü
    sample_path = os.path.join(base_dir, "processed_128", "train", "AnnualCrop",
                              random.choice(os.listdir(os.path.join(base_dir, "processed_128", "train", "AnnualCrop"))))
    
    with Image.open(sample_path) as img:
        print(f"Örnek Görüntü Boyutu: {img.size} → Beklenen: (128, 128)")
        print(f"Piksel Aralığı: {img.getextrema()} → Beklenen: (0-255)")

    print("\n🔄 4. Data Augmentation Kontrolü")
    
    # Data augmentation varyasyon kontrolü
    train_images = [os.path.join(base_dir, "processed_128", "train", "AnnualCrop", f) 
                   for f in os.listdir(os.path.join(base_dir, "processed_128", "train", "AnnualCrop"))]
    
    original_image = Image.open(os.path.join(base_dir, "Data", "Train", "AnnualCrop", 
                                           os.path.basename(train_images[0])))
    processed_image = Image.open(train_images[0])
    
    print(f"Aynı görüntünün ham vs. işlenmiş boyutu: {original_image.size} → {processed_image.size}")
    print("Manuel kontrol için 3 örnek görüntü gösteriliyor...")
    
    # Rastgele 3 görüntüyü gösterme
    for _ in range(3):
        img_path = random.choice(train_images)
        Image.open(img_path).show()

if __name__ == "__main__":
    check_data_integrity()