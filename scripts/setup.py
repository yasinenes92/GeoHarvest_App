import pkg_resources
import subprocess
import sys

# Gerekli kütüphaneler ve istenen sürümleri
required_packages = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "Pillow": "11.1.0",
    "numpy": "2.2.3",
    "opencv-python": "4.7.0.68",  # Güncellenmiş uyumlu sürüm
    "albumentations": "1.3.1"
}




def install_package(package, version):
    """Belirtilen paketi istenen sürümle yükler."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} yüklenirken hata oluştu: {e}")
        sys.exit(1)

def check_and_install_packages():
    """Gerekli paketlerin yüklü ve güncel olup olmadığını kontrol eder; değilse yükler."""
    for package, required_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if installed_version != required_version:
                print(f"⚠ {package} yüklü: {installed_version} (gerekli: {required_version}). Güncellenecek...")
                install_package(package, required_version)
            else:
                print(f"✔ {package} yüklü ve güncel: {installed_version}")
        except pkg_resources.DistributionNotFound:
            print(f"⚠ {package} yüklü değil. Yükleniyor ({required_version})...")
            install_package(package, required_version)

if __name__ == "__main__":
    check_and_install_packages()
    print("✅ Tüm gerekli kütüphaneler yüklü ve güncel!")
