import torch
from torchvision import models
import torch.nn as nn

def load_model(model_path):
    """
    Belirtilen yoldaki model ağırlıklarını kullanarak ResNet-50 tabanlı modeli yükler.
    Modelin son katmanı, iki sınıfı (AnnualCrop, HerbaceousVegetation) ayırt etmek için yeniden yapılandırılır.
    """
    # Önceden eğitilmiş ResNet-50'yi kullanmadan oluşturuyoruz
    model = models.resnet50(pretrained=False)
    # Modelin son fully-connected katmanının giriş boyutunu alıyoruz
    num_features = model.fc.in_features
    # Son katmanı, 2 çıkış verecek şekilde değiştiriyoruz
    model.fc = nn.Linear(num_features, 2)
    # Ağırlıkları belirtilen dosyadan yüklüyoruz (cpu üzerinden)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # Modeli değerlendirme moduna alıyoruz (dropout, batch norm gibi katmanların davranışı sabitlenir)
    model.eval()
    return model

# Örnek kullanım (dosyayı test etmek isterseniz, bu kısmı yorum satırına alabilirsiniz)
if __name__ == "__main__":
    model_path = "../model/best_model.pth"  # Eğer scripts klasöründeyseniz model klasörüne gitmek için "../" ekleyin
    model = load_model(model_path)
    print("Model başarıyla yüklendi!")
