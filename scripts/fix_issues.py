# fix_issues.py
import os
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt

# 1. YOL TANIMLAMALARI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "Data", "Train", "AnnualCrop", os.listdir(os.path.join(BASE_DIR, "Data", "Train", "AnnualCrop"))[0])

# 2. NORMALİZASYON KONTROL FONKSİYONU
def check_normalization():
    print("\n=== NORMALİZASYON KONTROLU ===")
    sample_path = os.path.join(BASE_DIR, "processed_128", "train", "AnnualCrop", 
                             os.listdir(os.path.join(BASE_DIR, "processed_128", "train", "AnnualCrop"))[0])
    
    with Image.open(sample_path) as img:
        img_array = np.array(img)
        print(f"Piksel Değer Aralığı: {img_array.min()} - {img_array.max()} (Beklenen: 0-255)")
        
    if img_array.max() > 1.0:
        print("⚠ UYARI: Model eğitiminde tekrar normalize etmeyi unutmayın!")
        print("Çözüm: model.transform'a Normalize() ekleyin (örnek kod yorum satırlarında)")

# 3. VERTICAL FLIP TEST FONKSİYONU
def test_vertical_flip():
    print("\n=== DIKEY ÇEVİRME TESTİ ===")
    original_img = np.array(Image.open(TEST_IMAGE_PATH))
    
    # Test için %100 olasılıkla dikey çevirme
    test_transform = A.Compose([
        A.Resize(128, 128),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformed = test_transform(image=original_img)["image"]
    transformed = (transformed * 255).astype(np.uint8)  # Orijinal kodunuzla uyumlu
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Orijinal Görüntü")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title("Dikey Çevrilmiş")
    plt.imshow(transformed)
    plt.axis('off')
    plt.show()

# 4. PROBLEMLERİ ÇÖZME FONKSİYONU
def fix_issues():
    print("\n=== SORUNLARI ÇÖZME ===")
    # 4.1 Normalizasyon Düzeltme
    print("Adım 1: Normalizasyon ayarı kontrol ediliyor...")
    check_normalization()
    
    # 4.2 Vertical Flip Düzeltme
    print("\nAdım 2: Dikey çevirme test ediliyor...")
    test_vertical_flip()
    
    # 4.3 Tüm veriyi yeniden işleme (opsiyonel)
    if input("\nTüm veriyi yeniden işlemek istiyor musunuz? (e/h): ").lower() == "e":
        from data_preprocessing import process_images, INPUT_PATHS, OUTPUT_PATHS
        print("\n⚠ DİKKAT: Bu işlem tüm veriyi yeniden oluşturacak!")
        
        # VerticalFlip p=0.5 ile orijinal ayara geri dön
        transforms = {
            "train": A.Compose([
                A.Resize(128, 128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "test": A.Compose([
                A.Resize(128, 128),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        
        process_images(INPUT_PATHS["train"], OUTPUT_PATHS["train"], transforms["train"])
        process_images(INPUT_PATHS["test"], OUTPUT_PATHS["test"], transforms["test"])
        print("\n✔ Veri başarıyla yeniden işlendi!")

if __name__ == "__main__":
    print("=== GEOHARVEST AI SORUN ÇÖZÜCÜ ===")
    fix_issues()
    print("\n⚠ MODEL EĞİTİMİNDE YAPILACAKLAR:")
    print("- Görüntüleri yüklerken Normalize() transformu ekleyin:")
    print("""from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])""")