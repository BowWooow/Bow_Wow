# Bow_Wow

### [PPT 발표자료](https://docs.google.com/presentation/d/1ik2TcG4YPUJjt2AelKIowR7Kf3dMm32ql4U0TN5m8eE/edit?usp=sharing)

### [GPU 사용설정 방법](https://oceanlightai.tistory.com/26)
┖ 사용 Python v3.9


저희의 코드는 합성곱 신경망(CNN : Convolutional Neural Network) 딥 러닝 모델을 사용하여 이미지를 분류하는 과정을 나타내고 있습니다. 
해당 코드는 특정 디렉토리에 있는 이미지 데이터를 기반으로 CNN(Convolutional Neural Network)을 구축하고, 훈련 및 테스트 데이터를 생성하여 모델을 훈련하고 평가합니다. 
추가적으로 사용자가 입력한 이미지를 분류하고 분류 결과를 출력하며, 훈련 과정에서의 손실과 정확도를 그래프로 시각화하는 부분까지 포함하고 있습니다.

코드에 대한 개요는 다음과 같습니다.
1. 이미지 데이터 디렉토리 경로 설정: 훈련 및 테스트 이미지 데이터 디렉토리 경로를 설정합니다.
2. 데이터 전처리 및 증강: 훈련 데이터를 위해 이미지 데이터 증강을 위한 ImageDataGenerator를 설정합니다.
3. 훈련 및 테스트 데이터 생성: ImageDataGenerator를 사용하여 훈련 및 테스트 데이터를 생성합니다.
4. CNN 모델 설계: Convolutional Neural Network(CNN) 모델을 Sequential 모델로 설계합니다.
5. 이미지 전처리 함수: 이미지를 모델에 입력할 수 있는 형태로 전처리하는 함수를 정의합니다.
6. 모델 가중치 불러오기: 미리 훈련된 모델 가중치를 불러옵니다.
7. 이미지 분류 및 출력: 사용자가 입력한 이미지를 분류하고 결과를 출력합니다.
8. 모델 훈련 및 평가: 모델을 훈련하고 평가하며, 훈련 과정에서의 손실과 정확도를 시각화합니다.

모델 훈련 및 평가를 수행하는 부분에서 사용자로부터 훈련 횟수를 입력받고 있으며, 모델의 정확도를 출력하고 있습니다. 
사용자로부터 이미지 파일명을 입력받아 해당 이미지를 분류하고, 분류 결과를 출력하고 있습니다. 
또한 훈련 과정에서의 손실과 정확도를 그래프로 시각화하여 표시하고 있습니다.

해당 코드는 TensorFlow와 Keras를 사용하여 CNN(Convolutional Neural Network) 모델을 훈련하고 이미지를 분류하는 작업을 수행합니다.

코드의 주요 부분을 요약하면 다음과 같습니다.
1. `ImageDataGenerator`를 사용하여 이미지 데이터를 증강하고 처리합니다.
2. CNN 모델을 Sequential 모델로 구축합니다. 이 모델은 Conv2D 레이어와 MaxPooling2D 레이어, 그리고 Dense 레이어로 구성되어 있습니다.
3. 훈련 및 테스트 데이터를 생성하기 위해 `flow_from_directory` 메서드를 사용하여 이미지 데이터를 불러옵니다.
4. 모델을 컴파일하고 이미지를 분류하기 위한 함수를 정의합니다.
5. 사용자에게 분류할 이미지 파일명을 입력 받습니다.
6. 입력된 이미지를 전처리하고, 모델을 사용하여 이미지를 분류하고 해당 결과를 출력합니다.
7. 사용자로부터 추가적인 훈련 횟수를 입력 받고, 모델을 이어서 훈련합니다.
8. 훈련 과정에서의 손실과 정확도를 기록하고, 그 결과를 시각화하여 표시합니다.

또한, 코드 실행을 위해서는 디렉토리 구조와 이미지 데이터가 적절히 준비되어 있어야 하며, TensorFlow 및 Keras 라이브러리가 설치되어 있어야 합니다. 
사용자로부터 입력을 받기 때문에 코드 실행 시 사용자의 입력을 기다리게 됩니다. 
코드 실행 전에 사용자 입력과 필요한 디렉토리 및 파일 경로들을 올바르게 설정해야 합니다.

또한, 코드 실행에 필요한 GPU가 있는 환경에서 실행될 것을 전제로 하고 있으며, GPU를 사용하기 위해 `tf.device('/GPU:0')`와 같은 코드를 사용하고 있습니다. 
해당 코드를 실행하기 위해서는 GPU 및 관련된 CUDA, cuDNN 등이 설치되어 있어야 하며, TensorFlow GPU 버전이 설치되어 있어야 합니다.
