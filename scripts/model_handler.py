import os
import sys
import torch
from torchvision import models
import torch.nn as nn
import logging

def get_model_path():
    """
    Proje dizin yapısına göre model dosyasının yolunu dinamik olarak belirler.
    Hem geliştirme hem de exe çalıştırma ortamları için uyumlu.
    """
    try:
        # PyInstaller ile paketlenmiş exe için
        if getattr(sys, 'frozen', False):
            # Exe ile aynı dizindeki model klasörü
            base_path = os.path.dirname(sys.executable)
            model_path = os.path.join(base_path, 'model', 'best_model.pth')
        else:
            # Normal Python çalıştırma için
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, "model", "best_model.pth")
        
        # Hata ayıklama bilgileri
        print(f"Model yolu: {model_path}")
        print(f"Model dosyası var mı? {os.path.exists(model_path)}")
        
        return model_path
    
    except Exception as e:
        # Hata günlüğü
        logging.error(f"Model yolu belirlenirken hata: {e}")
        raise

def load_model(model_path=None):
    """
    Belirtilen yoldaki model ağırlıklarını kullanarak ResNet-50 tabanlı modeli yükler.
    Parametre verilmezse otomatik olarak model yolu bulunur.
    """
    try:
        # Eğer model yolu verilmediyse otomatik bul
        if model_path is None:
            model_path = get_model_path()
        
        # Hata ayıklama bilgileri
        print(f"Model yükleniyor: {model_path}")
        print(f"Dosya boyutu: {os.path.getsize(model_path)} byte")
        
        # Model ağırlıklarını yükle
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Önceden eğitilmiş ResNet-50'yi kullanmadan oluşturuyoruz
        model = models.resnet50(pretrained=False)
        
        # Modelin son fully-connected katmanının giriş boyutunu alıyoruz
        num_features = model.fc.in_features
        
        # Son katmanı, 2 çıkış verecek şekilde değiştiriyoruz
        model.fc = nn.Linear(num_features, 2)
        
        # Yüklenmiş ağırlıkları modele atıyoruz
        model.load_state_dict(model_state_dict)
        
        # Modeli değerlendirme moduna alıyoruz
        model.eval()
        
        print("Model başarıyla yüklendi!")
        return model
    
    except Exception as e:
        # Detaylı hata günlüğü
        print(f"Model yükleme hatası: {e}")
        logging.error(f"Model yükleme hatası: {e}", exc_info=True)
        
        # Hata detaylarını içeren bir mesaj döndür
        raise RuntimeError(f"Model yüklenemedi: {e}")

# Örnek kullanım ve hata ayıklama
if __name__ == "__main__":
    try:
        # Logging ayarları
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='model_handler_log.txt'
        )
        
        # Model yükleme denemesi
        model = load_model()
        print("Model başarıyla yüklendi ve test edildi.")
    
    except Exception as e:
        print(f"Test sırasında hata oluştu: {e}")
        logging.error("Model yükleme testi başarısız", exc_info=True)