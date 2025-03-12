import sys
import os
import platform

def pyinstaller_diagnostics():
    print("🕵️ PyInstaller Tanılama Raporu 🕵️")
    
    # PyInstaller özel değişkenler
    print("\n1. PyInstaller Değişkenleri:")
    print(f"Frozen (Exe mi?): {getattr(sys, 'frozen', False)}")
    
    # Çalıştırılabilir dosya bilgileri
    if getattr(sys, 'frozen', False):
        print(f"Exe Yolu: {sys.executable}")
        print(f"Exe Dizini: {os.path.dirname(sys.executable)}")
    
    # Çalışma zamanı bilgileri
    print("\n2. Çalışma Zamanı Bilgileri:")
    print(f"Mevcut Çalışma Dizini: {os.getcwd()}")
    print(f"Script Dizini: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Sys.path içeriği
    print("\n3. Python Yol Listesi:")
    for path in sys.path:
        print(path)

if __name__ == "__main__":
    pyinstaller_diagnostics()