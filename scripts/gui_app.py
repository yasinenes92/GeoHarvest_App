#!/usr/bin/env python
"""
gui_app.py - GeoHarvest GUI Uygulaması

Bu script, kullanıcıların bir görsel seçip tahmin alabileceği basit bir Windows GUI oluşturur.
- "Görsel Seç" butonuyla kullanıcı, bir görsel dosyası seçer.
- Seçilen görsel, pencere içerisinde önizleme olarak gösterilir.
- "Tahmin Yap" butonuna basıldığında, model yüklenir, görsel ön işleme tabi tutulur, 
  model çalıştırılır ve tahmin sonucu (AnnualCrop veya HerbaceousVegetation) ekranda gösterilir.
"""

import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from model_handler import load_model      # Modeli yüklemek için oluşturduğumuz modül
from veri_on_isleme import preprocess_image  # Görseli ön işleme fonksiyonu

class GeoHarvestApp:
    def __init__(self, master):
        self.master = master
        self.master.title("GeoHarvest - Tarımsal Arazi Tanıma")
        
        # Proje dizin yapısı kullanılarak model dosyasının yolunu oluşturuyoruz.
        # Bu kod, "app" klasöründeyken projenin kök dizinine (GeoHarvest_App) çıkar ve "model" klasörünü hedefler.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "model", "best_model.pth")
        
        # Modeli yüklüyoruz ve float precision'a çeviriyoruz (model eğitiminde float32 kullanıldıysa)
        self.model = load_model(model_path)
        self.model = self.model.float()
        # Modelin tahmin ettiği sınıfların isimleri (eğitim sırasına göre ayarlayın)
        self.classes = ["AnnualCrop", "HerbaceousVegetation"]

        # GUI bileşenlerini oluşturuyoruz.
        self.label_info = Label(master, text="Görsel seçin ve tahmin yapın")
        self.label_info.pack(pady=10)
        
        self.btn_select = Button(master, text="Görsel Seç", command=self.select_file)
        self.btn_select.pack(pady=5)
        
        self.btn_predict = Button(master, text="Tahmin Yap", command=self.make_prediction, state=tk.DISABLED)
        self.btn_predict.pack(pady=5)
        
        self.label_image = Label(master)
        self.label_image.pack(pady=10)
        
        self.label_result = Label(master, text="")
        self.label_result.pack(pady=10)
        
        self.selected_file = None

    def select_file(self):
        """
        Kullanıcının dosya seçmesini sağlar.
        Seçilen dosyanın yolunu saklar, resmi yeniden boyutlandırarak pencereye önizleme olarak ekler.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.selected_file = file_path
            # Görseli aç, önizleme için 300x300 boyuta getir
            img = Image.open(file_path)
            img = img.resize((300, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.label_image.config(image=self.photo)
            self.btn_predict.config(state=tk.NORMAL)
            self.label_result.config(text="")

    def make_prediction(self):
        """
        Seçilen görseli alır, ön işleme tabi tutar, modeli çalıştırır ve tahmin sonucunu ekrana yazar.
        """
        if self.selected_file:
            try:
                # Görseli ön işleme fonksiyonuyla işleyip numpy array elde ediyoruz
                img_array = preprocess_image(self.selected_file)
                # Numpy array'i PyTorch tensörüne çeviriyoruz; batch boyutu ekleyip float32'e çeviriyoruz
                img_tensor = torch.tensor(img_array).unsqueeze(0).float()
                # Modeli tahmin modunda çalıştırıyoruz
                with torch.no_grad():
                    output = self.model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
                self.label_result.config(text=f"Tahmin: {self.classes[pred]}")
            except Exception as e:
                self.label_result.config(text=f"Hata: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GeoHarvestApp(root)
    root.mainloop()
