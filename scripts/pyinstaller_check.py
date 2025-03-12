import pkg_resources

def check_pyinstaller():
    try:
        # pkg_resources ile paket kontrolü
        pkg_resources.get_distribution('pyinstaller')
        print("✅ PyInstaller zaten yüklü")
        
        # Sürüm bilgisini de gösterelim
        import PyInstaller
        print(f"Sürüm: {PyInstaller.__version__}")
        return True
    
    except pkg_resources.DistributionNotFound:
        print("❌ PyInstaller yüklü değil")
        return False

if __name__ == "__main__":
    check_pyinstaller()