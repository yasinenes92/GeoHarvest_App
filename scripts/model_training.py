#!/usr/bin/env python
"""
model_training.py - 5. Gün: Learning Rate Schedule ve Early Stopping ile Eğitim

Bu script:
1. Eğitim ve test veri setlerini yükler (veri yolları proje yapınıza göre ayarlanır).
2. ResNet-50 modelini, pretrained ağırlıklar kullanarak oluşturur; ancak ilk 40 katmanı dondurulur ve son katman 2 sınıfa göre yeniden yapılandırılır.
3. Modeli belirlenen cihazda (GPU varsa 'cuda', yoksa 'cpu') çalıştırır.
4. 15 epoch boyunca eğitim yapar; her epoch sonunda eğitim ve doğrulama (validation) loss ve accuracy hesaplanır.
5. Learning rate, her 5 epoch'ta 0.1 katına düşürülür.
6. Doğrulama accuracy iyileşmezse early stopping devreye girer; en iyi model "model/best_model.pth" dosyasına kaydedilir.
7. Test aşamasında modelin performansı, classification report ve confusion matrix olarak kaydedilir.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import copy
from multiprocessing import freeze_support

def main():
    # 1. Cihaz Seçimi: GPU varsa 'cuda', yoksa 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)
    
    # 2. Veri Yolları
    # Bu script, GeoHarvest_App\scripts klasöründedir; proje kök dizini bir üst klasördedir.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_dir = os.path.join(project_root, "sample_images", "processed_128", "train")
    test_dir  = os.path.join(project_root, "sample_images", "processed_128", "test")
    
    # 3. Transform İşlemleri: Eğitim için augmentasyonlar, test için normalizasyon
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Dataset ve DataLoader
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 5. Model Tanımlaması
    # pretrained=True kullanarak ImageNet ağırlıklarını yüklüyoruz
    model = models.resnet50(pretrained=True)
    # İlk 40 katmanı donduruyoruz
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    # Son katmanı, 2 sınıfa (AnnualCrop, HerbaceousVegetation) göre yeniden tanımlıyoruz
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    # 6. Kayıp Fonksiyonu, Optimizer ve Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 7. Eğitim Döngüsü ve Early Stopping Ayarları
    num_epochs = 15
    best_accuracy = 0.0
    best_epoch = 0
    early_stop_patience = 3  # İyileşme olmazsa durdurma sayısı
    
    print("\nEğitime Başlanıyor...\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
        
        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects.double() / total_train
        print(f"Eğitim Loss: {epoch_loss:.4f} | Eğitim Acc: {epoch_acc:.4f}")
        
        # Doğrulama Aşaması
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
        
        val_loss = val_running_loss / total_val
        val_acc = val_running_corrects.double() / total_val
        print(f"Doğrulama Loss: {val_loss:.4f} | Doğrulama Acc: {val_acc:.4f}")
        
        # En iyi modeli kaydetme ve Early Stopping kontrolü
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            # Modeli, proje kök dizinindeki "model" klasörüne kaydediyoruz
            best_model_path = os.path.join(project_root, "model", "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Yeni en iyi model kaydedildi (Epoch {epoch+1})")
        else:
            if epoch - best_epoch >= early_stop_patience:
                print(f"🚨 Early Stopping (Epoch {epoch+1})")
                break
        
        scheduler.step()
        print()
    
    print(f"Eğitim tamamlandı. En iyi doğruluk: {best_accuracy:.4f}")
    
    # 8. Test Metriklerinin Hesaplanması
    model.load_state_dict(torch.load(os.path.join(project_root, "model", "best_model.pth")))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    class_names = test_dataset.classes
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    matrix = confusion_matrix(y_true, y_pred)
    
    # Classification Report ve Confusion Matrix'i dosyaya kaydediyoruz
    with open("classification_report.txt", "w") as f:
        f.write(report)
    with open("confusion_matrix.txt", "w") as f:
        f.write(np.array2string(matrix, separator=', '))
    
    print("✅ Metrikler kaydedildi!")
    
if __name__ == '__main__':
    freeze_support()  # Windows ortamında multiprocessing için gerekli olabilir.
    main()
