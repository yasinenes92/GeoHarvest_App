import os
import sys
import torch
import traceback

def detailed_model_loading():
    print("🔬 Detaylı Model Yükleme Tanılaması 🔬")
    
    # Mevcut çalışma dizini
    print(f"Mevcut Çalışma Dizini: {os.getcwd()}")
    
    # Olası model yolları - DİKKATLİ KONTROL EDİLDİ
    possible_paths = [
        # Proje kök dizininden model klasörüne giden yol
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "best_model.pth"),
        
        # Mutlak yol ile doğrudan model klasörü
        r"C:\Users\yasin\Desktop\GeoHarvest_AI\GeoHarvest_App\model\best_model.pth"
    ]

    for path in possible_paths:
        # Mutlak yolu normalize et
        normalized_path = os.path.normpath(path)
        print(f"\nDenenen Model Yolu: {normalized_path}")
        
        try:
            # Dosya varlığı kontrolü
            if not os.path.exists(normalized_path):
                print(f"❌ Dosya bulunamadı: {normalized_path}")
                continue

            # Dosya boyutu kontrolü
            file_size = os.path.getsize(normalized_path)
            print(f"Dosya Boyutu: {file_size} byte")

            # Model yükleme denemesi
            try:
                # Modeli yükleme
                model = torch.load(normalized_path, map_location=torch.device('cpu'))
                print("✅ Model başarıyla yüklendi!")
                
                # Model türünü ve detaylarını yazdırma
                print(f"\nModel Türü: {type(model)}")
                
                # Eğer model bir state_dict içeriyorsa detayları göster
                if isinstance(model, dict):
                    print("\nModel State Dict Detayları:")
                    for key, value in model.items():
                        print(f"{key}: {type(value)}")
                elif hasattr(model, 'state_dict'):
                    print("\nModel Katman Detayları:")
                    for name, param in model.state_dict().items():
                        print(f"{name}: {param.shape}")
                else:
                    print("Model state_dict metoduna sahip değil veya beklenen formatta değil.")

            except Exception as load_error:
                print(f"❌ Model yükleme hatası: {load_error}")
                print(traceback.format_exc())

        except Exception as path_error:
            print(f"Yol kontrolünde hata: {path_error}")

if __name__ == "__main__":
    detailed_model_loading()