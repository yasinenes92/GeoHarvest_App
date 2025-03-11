# -*- coding: utf-8 -*-
"""GeoHarvest AI - EuroSAT Veri Ön İşleme Scripti"""
import os
import numpy as np
from PIL import Image
import albumentations as A

# 1. YOL TANIMLAMALARI (Windows Uyumlu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Script'in bulunduğu dizin

INPUT_PATHS = {
    "train": os.path.join(BASE_DIR, "Data", "Train"),
    "test": os.path.join(BASE_DIR, "Data", "Test")
}

OUTPUT_PATHS = {
    "train": os.path.join(BASE_DIR, "processed_128", "Train"),
    "test": os.path.join(BASE_DIR, "processed_128", "Test")
}

# 2. DÖNÜŞÜM TANIMLAMALARI (Albumentations 0.5.2 ile test edildi)
transforms = {
    "train": A.Compose([
        A.Resize(128, 128, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": A.Compose([
        A.Resize(128, 128, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 3. İŞLEM FONKSİYONU
def process_images(input_dir, output_dir, transform):
    """Görüntüleri işler ve kaydeder"""
    try:
        # Klasör kontrolü
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"{input_dir} bulunamadı!")
        
        print(f"\n▶ {os.path.basename(input_dir)} işleniyor...")
        
        # Her sınıf için işlem
        for class_name in ["AnnualCrop", "HerbaceousVegetation"]:
            class_input_path = os.path.join(input_dir, class_name)
            class_output_path = os.path.join(output_dir, class_name)
            
            os.makedirs(class_output_path, exist_ok=True)
            
            # Görüntü işleme
            for img_name in os.listdir(class_input_path):
                img_path = os.path.join(class_input_path, img_name)
                
                # Dönüşüm uygula
                image = np.array(Image.open(img_path).convert("RGB"))
                transformed = transform(image=image)["image"]
                transformed = (transformed * 255).astype(np.uint8)
                
                # Kaydet
                output_path = os.path.join(class_output_path, img_name)
                Image.fromarray(transformed).save(output_path)
                
        print(f"✔ {os.path.basename(input_dir)} başarıyla işlendi!")
        return True
    
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        return False

# 4. ANA İŞLEM
if __name__ == "__main__":
    print("=== GEOHAREST AI VERİ ÖN İŞLEME ===")
    print(f"Ana dizin: {BASE_DIR}\n")
    
    # Tüm işlemleri çalıştır
    success = all([
        process_images(INPUT_PATHS["train"], OUTPUT_PATHS["train"], transforms["train"]),
        process_images(INPUT_PATHS["test"], OUTPUT_PATHS["test"], transforms["test"])
    ])
    
    # Sonuç kontrolü
    if success:
        try:
            sample_path = os.path.join(OUTPUT_PATHS["train"], "AnnualCrop", os.listdir(os.path.join(OUTPUT_PATHS["train"], "AnnualCrop"))[0])
            img = Image.open(sample_path)
            print("\n=== SONUÇ ===")
            print(f"Örnek görüntü boyutu: {img.size}")
            print("✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
        except:
            print("\n⚠ UYARI: Sonuç kontrolü yapılamadı!")
    else:
        print("\n❌ İŞLEMLER HATAYLA SONUÇLANDI!")