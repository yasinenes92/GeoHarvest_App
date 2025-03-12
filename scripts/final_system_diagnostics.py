import os
import sys
import torch
import tkinter as tk
from PIL import Image
import torchvision.transforms as transforms

def final_system_check():
    print("🔍 Final Sistem Kontrol Raporu 🔍")
    
    # Proje kök dizini
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Kritik dosyalar listesi
    critical_files = [
        os.path.join(project_root, 'model', 'best_model.pth'),
        os.path.join(project_root, 'scripts', 'gui_app.py'),
        os.path.join(project_root, 'scripts', 'model_handler.py'),
        os.path.join(project_root, 'scripts', 'veri_on_isleme.py')
    ]
    
    # Dosya kontrolleri
    print("\n1. Kritik Dosya Kontrolleri:")
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)} - Mevcut")
        else:
            print(f"❌ {os.path.basename(file_path)} - BULUNAMADI!")
    
    # Bağımlılık kontrolleri
    print("\n2. Kritik Bağımlılık Kontrolleri:")
    dependencies = [
        ('torch', torch.__version__),
        ('torchvision', torch.__version__),
        ('PIL', Image.__version__),
        ('tkinter', tk.TkVersion)
    ]
    
    for name, version in dependencies:
        print(f"✅ {name} - Sürüm: {version}")
    
    # Model yükleme testi
    print("\n3. Model Yükleme Testi:")
    try:
        from model_handler import load_model
        model = load_model()
        print("✅ Model başarıyla yüklendi")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
    
    # Ön işleme fonksiyon testi
    print("\n4. Ön İşleme Fonksiyon Testi:")
    try:
        from veri_on_isleme import preprocess_image
        
        # Test görseli yolunu dinamik olarak bul
        test_image_paths = [
            os.path.join(project_root, 'sample_images', 'processed_128', 'test', 'AnnualCrop', 'AnnualCrop_2401.jpg'),
            os.path.join(project_root, 'sample_images', 'processed_128', 'test', 'HerbaceousVegetation', 'HerbaceousVegetation_2401.jpg')
        ]
        
        # İlk bulunan test görselini kullan
        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if test_image_path:
            print(f"Test görseli: {test_image_path}")
            processed_image = preprocess_image(test_image_path)
            print("✅ Görüntü ön işleme başarılı")
            print(f"İşlenmiş görüntü şekli: {processed_image.shape}")
        else:
            print("❌ Test görseli bulunamadı")
    except Exception as e:
        print(f"❌ Ön işleme hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    final_system_check()