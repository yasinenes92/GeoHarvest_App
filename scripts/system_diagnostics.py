import os
import sys
import traceback
import logging

def comprehensive_diagnostics():
    # Sistem ve Çalışma Ortamı Bilgileri
    print("🔍 Kapsamlı Tanılama Raporu 🔍")
    print("\n1. Sistem Bilgileri:")
    print(f"Python Sürümü: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Exe mi?: {getattr(sys, 'frozen', False)}")

    # Dizin Yapısı Kontrolü
    print("\n2. Dizin Yapısı:")
    def list_directory(path):
        try:
            return os.listdir(path)
        except Exception as e:
            return f"Dizin okunamadı: {e}"

    base_paths = [
        os.getcwd(),  # Mevcut çalışma dizini
        os.path.dirname(sys.executable),  # Exe dizini
        os.path.dirname(os.path.abspath(__file__))  # Script dizini
    ]

    for path in base_paths:
        print(f"\nDizin: {path}")
        try:
            contents = list_directory(path)
            print("İçerik:", contents)
        except Exception as e:
            print(f"Dizin listelenemedi: {e}")

    # Model Dosyası Kontrolleri
    print("\n3. Model Dosyası Kontrolleri:")
    def check_model_file():
        # Olası model dosyası yolları - DÜZELTME YAPILDI
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "best_model.pth"),
            os.path.join(os.getcwd(), "..", "model", "best_model.pth"),
            os.path.join(os.path.dirname(sys.executable), "..", "..", "model", "best_model.pth")
        ]

        for path in possible_paths:
            # Mutlak yolu normalize et
            normalized_path = os.path.normpath(path)
            print(f"Kontrol edilen yol: {normalized_path}")
            try:
                if os.path.exists(normalized_path):
                    print(f"✅ Dosya bulundu!")
                    print(f"Dosya boyutu: {os.path.getsize(normalized_path)} byte")
                else:
                    print(f"❌ Dosya bulunamadı.")
            except Exception as e:
                print(f"Dosya kontrolünde hata: {e}")

    check_model_file()

    # Bağımlılık Kontrolleri
    print("\n4. Bağımlılık Kontrolleri:")
    try:
        import torch
        import torchvision
        print(f"PyTorch Sürümü: {torch.__version__}")
        print(f"TorchVision Sürümü: {torchvision.__version__}")
    except ImportError as e:
        print(f"Bağımlılık hatası: {e}")

    # Hata Günlüğü Oluşturma
    print("\n5. Hata Günlüğü:")
    logging.basicConfig(filename='diagnostics_log.txt', level=logging.DEBUG)
    try:
        # Olası hata senaryoları
        raise Exception("Tanılama testi")
    except Exception as e:
        logging.error("Tanılama hatası", exc_info=True)
        print("Hata günlüğü oluşturuldu: diagnostics_log.txt")

if __name__ == "__main__":
    comprehensive_diagnostics()