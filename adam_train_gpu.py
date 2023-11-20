import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from pathlib import Path

# GPU 디바이스 확인
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 모델이 GPU를 사용하도록 설정
with tf.device('/GPU:0'):
    
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

    batch_size = 128
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True  # 데이터를 섞어 빠르게 불러오기
    )

    # 테스트 데이터 생성
    batch_size = 128
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # 테스트 데이터는 섞을 필요 없음
    )

    # 모델 아키텍처 설계 (건들면 작동안함)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(120, activation='softmax')) # 120은 클래스 수

    # 모델 가중치 불러오기
    model.load_weights(r'C:\Users\Mss\Desktop\project\ai\content\model_weights_adam.h5')

    # 모델 컴파일
    model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # 모델 훈련
    epoch = int(input("훈련 횟수 : "))
    history = model.fit(train_generator, epochs=epoch, validation_data=test_generator)

    # 모델 가중치를 H5 파일로 저장
    model.save_weights(r'C:\Users\Mss\Desktop\project\ai\content\model_weights_adam.h5')

    # 모델 평가
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"테스트 정확도: {test_acc}")

    # 이미지 데이터 디렉토리 경로
    data_dir = Path("C:/Users/Mss/Desktop/project/ai/content/train")

    # 클래스 디렉토리 목록
    class_directories = list(data_dir.glob("*"))

    # 클래스 디렉토리 및 클래스 이름 압축
    class_directory_mapping = {dir.name: dir for dir in class_directories}
    class_names = list(class_directory_mapping.keys())

    # 무작위로 클래스 선택
    selected_class = random.choice(list(class_directory_mapping.keys()))
    image_directory = class_directory_mapping[selected_class]

    # 출력
    print(f"선택된 클래스: {selected_class}")
    print(f"디렉토리 경로: {image_directory}")

    # 무작위 이미지 선택
    image_files = list(image_directory.glob("*.jpg"))
    selected_image_path = random.choice(image_files)
    file_name = os.path.basename(selected_image_path)

    def load_and_preprocess_image(image_path, target_size=(128, 128)):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0  # 정규화
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # 이미지를 모델 입력 크기로 조정
    img = load_and_preprocess_image(selected_image_path)

    # 클래스 인덱스 가져오기
    predicted_class_index = np.argmax(model.predict(img))

    # 분류된 클래스 출력
    predicted_class = class_names[predicted_class_index]

    print(f"이미지 {file_name} 는 {predicted_class}로 분류됩니다.")
