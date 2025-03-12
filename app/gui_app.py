#!/usr/bin/env python
"""
gui_app.py - GeoHarvest GUI Uygulaması

Bu script, kullanıcıların bir görsel seçip tahmin alabileceği basit bir Windows GUI oluşturur.
- "Görsel Seç" butonuyla kullanıcı, bir görsel dosyası seçer.
- Seçilen görsel, pencere içerisinde önizleme olarak gösterilir.
- "Tahmin Yap" butonuna basıldığında, model yüklenir, görsel ön işleme tabi tutulur,
  model çalıştırılır ve tahmin sonucu (AnnualCrop veya HerbaceousVegetation) ekranda gösterilir.
"""

import sys
import os

# Frozen (exe) modunda çalışıyorsak, sys._MEIPASS dizinini kullanıyoruz.
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Proje kök dizinini ayarlıyoruz.
if getattr(sys, 'frozen', False):
    # Eğer paketlenmişse, varsayılan olarak base_path proje kök dizini olarak kabul edilir.
    project_root = base_path
else:
    # Normal çalıştırmada, gui_app.py "app" klasöründe; kök dizin bir üst klasördür.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# "scripts" klasörünün yolunu oluşturup sys.path'e ekleyelim.
scripts_path = os.path.join(project_root, "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from model_handler import load_model      # "scripts" klasöründeki model_handler.py
from veri_on_isleme import preprocess_image  # "scripts" klasöründeki veri_on_isleme.py

class GeoHarvestApp:
    def __init__(self, master):
        self.master = master
        self.master.title("GeoHarvest - Tarımsal Arazi Tanıma")
        
        # Model dosyasının yolunu, proje kök dizininden "model" klasörüne göre oluşturuyoruz.
        model_path = os.path.join(project_root, "model", "best_model.pth")
        
        # Modeli yüklüyoruz ve float precision'a çeviriyoruz.
        self.model = load_model(model_path)
        self.model = self.model.float()
        # Modelin tahmin ettiği sınıf isimleri (eğitim sırasına göre ayarlayın)
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
                img_array = preprocess_image(self.selected_file)
                img_tensor = torch.tensor(img_array).unsqueeze(0).float()
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
