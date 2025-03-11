# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# GPU Kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")

# Veri Yolları
train_dir = "/content/drive/MyDrive/processed_128/train"
test_dir = "/content/drive/MyDrive/processed_128/test"

# Transform İşlemleri
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2)

# Model Tanımı
model = models.resnet50(pretrained=True)
for param in list(model.parameters())[:-10]:
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Early Stopping
patience = 3
best_accuracy = 0.0
best_epoch = 0

# Eğitim Döngüsü
for epoch in range(15):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    
    train_loss = train_loss / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (preds == labels).sum().item()
    
    val_loss = val_loss / len(test_dataset)
    val_accuracy = correct / len(test_dataset)
    
    print(f"Epoch {epoch+1}/15 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    # Early Stopping ve Model Kaydetme
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✔ Yeni en iyi model (Epoch {epoch+1})")
    elif epoch - best_epoch >= patience:
        print(f"🚨 Early Stopping (Epoch {epoch+1})")
        break
    
    scheduler.step()

# Test Metrikleri
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Classification Report ve Confusion Matrix Kaydet
class_names = test_dataset.classes
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
matrix = confusion_matrix(y_true, y_pred)

with open("classification_report.txt", "w") as f:
    f.write(report)

with open("confusion_matrix.txt", "w") as f:
    f.write(np.array2string(matrix, separator=', '))

print("✅ Metrikler kaydedildi!")