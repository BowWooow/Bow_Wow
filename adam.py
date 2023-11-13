import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import layers, models
from pathlib import Path

# 이미지 데이터 디렉토리 경로 설정
current_dir = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(current_dir, "content/train")
test_dir = os.path.join(current_dir, "content/test")

# 데이터 전처리 및 증강
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# 훈련 데이터 생성
batch_size = 64
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # 데이터를 섞어 빠르게 불러오기
)

# 테스트 데이터 생성
batch_size = 64
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # 테스트 데이터는 섞을 필요 없음
)

# 모델 아키텍처 설계
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(120, activation='softmax'))  # 120은 클래스 수

# 이미지 데이터 디렉토리 경로
data_dir = Path(os.path.join(current_dir, "content/train"))

# 클래스 디렉토리 목록
class_directories = list(data_dir.glob("*"))

# 클래스 디렉토리 및 클래스 이름 압축
class_directory_mapping = {dir.name: dir for dir in class_directories}
class_names = list(class_directory_mapping.keys())


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 모델 가중치 불러오기
model.load_weights(os.path.join(current_dir, 'model_weights_adam.h5'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 이미지 선택
external_image_name = input("분류할 이미지 (확장자 포함) : ")
external_image_path = os.path.join(current_dir, "content/img_test", external_image_name)

# 이미지를 모델 입력 크기로 조정
img = load_and_preprocess_image(external_image_path)

# 클래스 인덱스 가져오기
predicted_class_index = np.argmax(model.predict(img))


# 모델 훈련
epoc = input("훈련 횟수 : ")
epoch = int(epoc)
history = model.fit(train_generator, epochs=epoch, validation_data=test_generator)

# 모델 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"테스트 정확도: {test_acc}")

# 분류된 클래스 출력
predicted_class = list(class_names)[predicted_class_index]

print(f"이미지{external_image_name}는 {predicted_class}로 분류됩니다.")

# 훈련 과정에서의 손실과 정확도
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# 그래프
fig, ax1 = plt.subplots()

# 손실 그래프
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(range(1, epoch + 1), train_loss, label='Train Loss', color='tab:red')
ax1.plot(range(1, epoch + 1), val_loss, label='Validation Loss', color='tab:orange')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')

# 정확도 그래프
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:blue')
ax2.plot(range(1, epoch + 1), train_acc, label='Train Accuracy', color='tab:blue')
ax2.plot(range(1, epoch + 1), val_acc, label='Validation Accuracy', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

# 그래프 저장
combined_fig_path = os.path.join(current_dir, 'combined_graph.png')
plt.savefig(combined_fig_path)
plt.show()
