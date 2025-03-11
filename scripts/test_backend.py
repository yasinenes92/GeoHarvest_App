import os
import torch
from model_handler import load_model
from veri_on_isleme import preprocess_image

def test_model_and_preprocessing():
    # __file__ ile dosyanın bulunduğu dizini alın
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Model dosyasının yolu: scripts klasöründen bir üst klasöre çıkıp model klasörüne git
    model_path = os.path.join(current_dir, "..", "model", "best_model.pth")
    model = load_model(model_path)
    # Modeli float precision'a dönüştürme (eğitimde float32 kullanıldıysa)
    model = model.float()
    print("Model başarıyla yüklendi.")

    # Sample görselin yolu: scripts klasöründen bir üst klasöre çıkıp sample_images/processed_128/train/AnnualCrop
    sample_image_path = os.path.join(current_dir, "..", "sample_images", "processed_128", "train", "AnnualCrop", "AnnualCrop_1.jpg")
    
    print("Çalışma dizini:", os.getcwd())
    print("Örnek görsel yolu:", sample_image_path)
    
    try:
        img_array = preprocess_image(sample_image_path)
    except Exception as e:
        print("Ön işleme hatası:", e)
        return

    # Numpy array'i PyTorch tensörüne çeviriyoruz (batch boyutu ekleyerek)
    # Giriş tensörünü float32'e dönüştürüyoruz
    img_tensor = torch.tensor(img_array).unsqueeze(0).float()
    print("Görsel başarıyla ön işlendi ve tensor formatına dönüştürüldü. Tensor boyutu:", img_tensor.shape)
    print("Giriş tensörünün veri tipi:", img_tensor.dtype)
    print("Modelin ilk katmanının ağırlık tipi:", next(model.parameters()).dtype)

    # Modeli çalıştırıp tahmin yapmayı deneyelim
    with torch.no_grad():
        output = model(img_tensor)
    pred = torch.argmax(output, dim=1).item()
    classes = ["AnnualCrop", "HerbaceousVegetation"]
    print("Tahmin:", classes[pred])

if __name__ == "__main__":
    test_model_and_preprocessing()
