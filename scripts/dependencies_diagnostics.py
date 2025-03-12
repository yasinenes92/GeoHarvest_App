import sys
import os
import platform

def check_dependencies():
    print("🔍 Bağımlılık ve Sistem Kontrol Raporu 🔍")
    
    # Sistem Bilgileri
    print("\n1. Sistem Bilgileri:")
    print(f"İşletim Sistemi: {platform.system()}")
    print(f"İşletim Sistemi Sürümü: {platform.version()}")
    print(f"Python Sürümü: {sys.version}")
    
    # Bağımlılık Kontrolleri
    print("\n2. Bağımlılık Kontrolleri:")
    dependencies = [
        'torch', 
        'torchvision', 
        'tkinter', 
        'PIL'
    ]
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            print(f"{dep}: ✅ Yüklü (Sürüm: {module.__version__ if hasattr(module, '__version__') else 'Bilinmiyor'})")
        except ImportError:
            print(f"{dep}: ❌ Yüklenemedi")
    
    # Çalışma Dizini Kontrolleri
    print("\n3. Dizin Kontrolleri:")
    print(f"Mevcut Çalışma Dizini: {os.getcwd()}")
    print(f"Script Dizini: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Proje Dizin Yapısı Kontrolü
    print("\n4. Proje Dizin Yapısı:")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_dirs = ['model', 'scripts', 'sample_images']
    
    for directory in expected_dirs:
        dir_path = os.path.join(project_root, directory)
        if os.path.exists(dir_path):
            print(f"{directory} dizini: ✅ Mevcut")
            print(f"  İçerik: {os.listdir(dir_path)}")
        else:
            print(f"{directory} dizini: ❌ Bulunamadı")

if __name__ == "__main__":
    check_dependencies()