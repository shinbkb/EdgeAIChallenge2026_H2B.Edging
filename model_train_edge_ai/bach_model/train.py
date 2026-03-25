
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

train_dir = os.path.join("warmup", "kaggle_testing", "train")

# 1. Dây chuyền dùng lúc HUẤN LUYỆN
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. Dây chuyền dùng lúc DỰ ĐOÁN TẬP TEST
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Đã sẵn sàng huấn luyện với {len(train_dataset)} ảnh!")


class MicroTrafficSignNet(nn.Module):
    def __init__(self):
        super(MicroTrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Đang chạy trên thiết bị: {device}")

model = MicroTrafficSignNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# PHẦN 5: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
epochs = 40
print("Bắt đầu quá trình huấn luyện...")

best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = (correct / total) * 100
    print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # ĐÃ SỬA: Lùi vào trong vòng lặp for. Mỗi vòng lặp xong đều kiểm tra xem có phá kỷ lục không!
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_micro_model.pth")
        print(f"   -> 🌟 Kỷ lục mới! Đã lưu best_micro_model.pth")

print("🎉 Huấn luyện hoàn tất!")

print("\nBắt đầu chạy dự đoán trên tập test...")

test_dir = os.path.join("warmup", "kaggle_testing", "test")


class KaggleTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, file_name


# Khởi tạo DataLoader cho Test
test_dataset = KaggleTestDataset(root_dir=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# LOAD MÔ HÌNH XỊN NHẤT TRƯỚC KHI DỰ ĐOÁN
model = MicroTrafficSignNet().to(device)
vi_tri_file = "best_micro_model.pth"

# Kiểm tra xem file đã tồn tại chưa (tránh lỗi nếu lỡ tắt ngang quá trình train)
if os.path.exists(vi_tri_file):
    trang_thai_da_luu = torch.load(vi_tri_file, map_location=device, weights_only=True)
    model.load_state_dict(trang_thai_da_luu)
    print(f"📥 Đã load bộ não xịn nhất từ {vi_tri_file}")
else:
    print(f"⚠️ Không tìm thấy {vi_tri_file}! Sẽ dự đoán bằng mô hình hiện tại trên RAM.")

# Bật chế độ đi thi
model.eval()
results = []

# Bắt đầu vòng lặp dự đoán
with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        for file_name, pred_class in zip(filenames, predicted):
            clean_id = file_name.split('.')[0]
            results.append({
                "Id": clean_id,
                "label": pred_class.item()
            })

df = pd.DataFrame(results)
output_csv = "submission.csv"
df.to_csv(output_csv, index=False)

print(f"✅ Đã dự đoán hoàn tất {len(results)} ảnh!")
print(f"📁 File nộp bài đã được lưu tại: {os.path.abspath(output_csv)}")