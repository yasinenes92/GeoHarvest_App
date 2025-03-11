#!/usr/bin/env python
"""
predict_cli.py - Basit Komut Satırı Arayüzü ile Tahmin Yapma

Bu script, kullanıcıdan girilen görsel dosya yolunu alır, görseli ön işler,
eğitilmiş modelimizi (best_model.pth) yükler ve modelin tahminini ekrana yazdırır.
"""

import sys
import os
import torch
from model_handler import load_model
from veri_on_isleme import preprocess_image  # Ön işleme fonksiyonunu içeren modül

def predict(image_path, model):
    """
    Verilen görsel yolundaki resmi ön işler, tensor formatına dönüştürür,
    modeli çalıştırır ve tahmin sonucunu döndürür.
    """
    # Görseli ön işleme fonksiyonuyla işleyip numpy array elde ediyoruz
    img_array = preprocess_image(image_path)
    # Array'i PyTorch tensörüne çeviriyoruz; batch boyutu ekleyip float32'e çeviriyoruz
    img_tensor = torch.tensor(img_array).unsqueeze(0).float()
    # Modeli tahmin modunda çalıştırıyoruz
    with torch.no_grad():
        output = model(img_tensor)
    # Tahmini belirlemek için en yüksek çıktıyı veren sınıfı seçiyoruz
    prediction = torch.argmax(output, dim=1).item()
    return prediction

def main():
    # Kullanıcıdan görsel dosya yolunu parametre olarak alıyoruz
    if len(sys.argv) != 2:
        print("Kullanım: python predict_cli.py <görsel_yolu>")
        sys.exit(1)
    image_path = sys.argv[1]

    # __file__ ile bulunduğunuz dizini alıyoruz
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Proje kök dizini; scripts klasörünün bir üstü
    project_root = os.path.dirname(current_dir)
    
    # Eğer girilen yol absolute değilse, proje kök dizinine göre oluşturuyoruz.
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)
    
    # Model dosyasının yolu: proje kök dizininden model klasörüne
    model_path = os.path.join(project_root, "model", "best_model.pth")
    
    # Modeli yüklüyoruz
    model = load_model(model_path)
    # Modelin doğru precision'da çalıştığından emin olmak için float'a çeviriyoruz
    model = model.float()
    
    # Tahmini alıyoruz
    pred = predict(image_path, model)
    # Sınıf isimlerini tanımlıyoruz (model eğitiminizde kullandığınız sıraya göre)
    classes = ["AnnualCrop", "HerbaceousVegetation"]
    print(f"Görselin tahmini: {classes[pred]}")

if __name__ == "__main__":
    main()
