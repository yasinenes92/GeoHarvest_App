from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Verilen dosya yolundaki resmi açar, 128x128 boyutuna yeniden boyutlandırır,
    ImageNet normalize değerleri kullanılarak normalize eder ve numpy dizisine çevirir.
    Kanal sırası HWC'den CHW'ye dönüştürülür.
    """
    # Resmi aç ve RGB'ye dönüştür
    img = Image.open(image_path).convert("RGB")
    # Yeniden boyutlandırma: 128x128 piksel
    img = img.resize((128, 128))
    # Resmi numpy array'e çevir (float32 ve [0,1] aralığında)
    img_array = np.array(img).astype('float32') / 255.0
    # Normalize et: ImageNet ortalamaları ve standart sapmaları
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    # Kanal sırasını HWC'den CHW'ye dönüştür (PyTorch'un beklediği format)
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array

# Örnek kullanım (test etmek için, bu kısmı yorum satırına alabilirsiniz)
if __name__ == "__main__":
    sample_image_path = "../sample_images/processed_128/train/AnnualCrop/some_sample.jpg"  # Örnek görsel yolu; uygun şekilde düzenleyin.
    processed = preprocess_image(sample_image_path)
    print("Ön işleme tamamlandı, array boyutu:", processed.shape)
