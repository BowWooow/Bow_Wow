import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import os
from pathlib import Path
from keras.preprocessing.image import load_img, img_to_array

# 이미지 데이터 디렉토리 경로 설정
current_dir = Path(__file__).parent
train_dir = current_dir / "content/train"
test_dir = current_dir / "content/test"

# 데이터 전처리 및 증강
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 훈련 및 테스트 데이터셋 생성
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

# 데이터로더 생성
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 아키텍처 설계
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 클래스 디렉토리 목록
class_directories = list(test_dir.glob("*"))

# 클래스 디렉토리 및 클래스 이름 압축
class_directory_mapping = {dir.name: dir for dir in class_directories}
class_names = list(class_directory_mapping.keys())


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 모델 및 손실 함수, 옵티마이저 생성
model = CustomModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 이미지 선택
external_image_name = input("분류할 이미지 (확장자 포함) : ")
external_image_path = current_dir / "content/img_test" / external_image_name

# 이미지를 모델 입력 크기로 조정
img = load_and_preprocess_image(external_image_path)
img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
img_tensor = img_tensor.permute(0, 3, 1, 2)

# 클래스 인덱스 가져오기
model.eval()
with torch.no_grad():
    output = model(img_tensor)
predicted_class_index = torch.argmax(output).item()

# 훈련 및 평가
num_epochs = 10
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    # 모델 평가
    model.eval()
    correct_val = 0
    total_val = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_loss = val_running_loss / len(test_loader)
    val_acc = correct_val / total_val
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 에폭마다 출력
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# 모델 평가 결과 출력
print(f"최종 테스트 정확도: {val_acc}")

# 분류된 클래스 출력
class_names = test_dataset.classes
predicted_class = class_names[predicted_class_index]

print(f"이미지 {external_image_name}는 {predicted_class}로 분류됩니다.")