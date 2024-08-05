# 신경망 학습

## 프로젝트 소개
Fashion Mnist Data를 학습시켜서 이미지에서 패션 카테고리 추출

## 문제 소개
10가지 패션 아이템 카테고리로 분류된 이미지로 구성된 Fashion-MNIST 데이터셋을 사용하여 MLP(Multi-Layer Perceptron) 모델을 학습하는 것을 목표로 한다. 학습된 MLP 모델은 제시된 이미지의 패션 카테고리를 추출하는 데 활용된다.

## 활용 DATA
![image](https://github.com/user-attachments/assets/2b63a77b-1131-426a-b48e-44848136c363)
<p align="center"><i>MNIST Fashion Data Set</i></p>

10개의 패션 아이템 카테고리를 가지고 있는 fashion 데이터 set. 60,000개의 훈련 이미지와 10,000개의 테스트 이미지로 총 70,000개의 회색조 이미지로 구성되어 있으며 각각의 해상도는 28*28 픽셀로 이루어져 있다.

## 모델 학습 단계
![image](https://github.com/user-attachments/assets/01a06137-7ec5-4a0d-8346-baff9463836c)

-	데이터 전처리: 이미지를 28x28 크기로 변환하고, 각 픽셀 값을 0에서 1 사이의 값으로 정규화한다.
-	데이터 분할: 데이터 세트를 학습 세트(60,000개)와 테스트 세트(10,000개)로 분할한다.
-	MLP 모델 학습: 정규화된 데이터를 입력으로 하여 MLP 모델을 학습시킨다. MLP 모델은 여러 개의 은닉층을 포함하며, 각 층은 ReLU 활성화 함수를 사용한다. 과적합을 방지하기 위해 Dropout 기법을 적용한다.
-	모델 평가: 테스트 데이터를 사용하여 모델의 성능을 평가하고, 각 패션 아이템 카테고리에 대한 분류 정확도를 측정한다.

## 모델 적합도 평가
![image](https://github.com/user-attachments/assets/eeb5efec-101b-4703-8fca-df3e5689c610)
<p align="center"><i>뉴런 별 학습 손실률</i></p>

![image](https://github.com/user-attachments/assets/a810552a-5a33-43fc-b56a-f6b879c219f8)
<p align="center"><i>카테고리 별 학습 정확도</i></p>

![image](https://github.com/user-attachments/assets/39a0e816-76dc-4a1a-b03b-6ae5c5af2957)
<p align="center"><i>모델 패션 카테고리 분석 결과</i></p>

![image](https://github.com/user-attachments/assets/db4ff418-27ab-4c69-8193-150facee7aad)
<p align="center"><i>개선 전 모델 정확도</i></p>

![image](https://github.com/user-attachments/assets/cde59671-70a4-4e4b-a4b9-99933be10bd6)
<p align="center"><i>개선 후 모델 정확도</i></p>

## 추가 개선사항

![image](https://github.com/user-attachments/assets/b845807c-f1d6-49f9-a1dc-553dc6a01bca)
<p align="center"><i>슬라이딩 윈도우 알고리즘</i></p>

입력 받은 이미지에서 패션 카테고리를 추출하기 위해  객체 감지를 위한  DPM(Deformable Parts Model) 방법을 기반으로  슬라이딩 윈도우 알고리즘을 실험했다.(RNN, CNN을 사용하지 않고 구현하기 위함) 슬라이딩 윈도우 알고리즘은 고정된 크기의 그리드를 사용하여 전체 이미지를 체계적으로 스캔하고 DPM 모델을 적용하여 각 그리드 내의 개체를 감지한다. 슬라이딩 윈도우 알고리즘은 실용적이고 간단하므로 특정 요구 사항과 리소스 제약이 있는 프로젝트에 적합하다고 생각한다.

![image](https://github.com/user-attachments/assets/6592f82b-8bec-4120-a6e9-be8de3439d22)
<p align="center"><i>객체 검출 방식 변경</i></p>

## 코드 실행 전 모듈 설치

1. pip install numpy
2. pip install opencv-python
3. pip install tensorflow
4. pip install matplotlib
5. pip install scikit-learn
6. pip install os-sys
7. pip install keras

## 코드 수행 절차

1. img 폴더에 사진 파일 업로드
2. tensorflow.keras 라이브러리를 인식하지 못할 경우
   window: python -m venv myenv
   Mac: python3 -m venv myenv
4. main.py 실행
5. mlp.py에서 learning 함수를 호출하여 fashion mnist data set을 불러와서 학습시킨다.
6. 모델 테스트 결과는 img/result에 이미지로 생성
7. fashion_mnist_mlp.h5 파일이 생성되어야 해당 분석이 가능
8. data set 학습 완료 후 predict.py에서 predict_image 함수를 호출하여 학습된 모델을 바탕으로 경로에 있는 이미지에서 패션 카테고리 추출.
9. 분석 결과는 img/result에 이미지로 생성
10. 위 모든 사항을 진행시에도 코드가 실행이 안될경우 제공된 fashion.ipynb 파일을 통해 colab에서 실행
### main.py

from mlp import learning
from predict import predict_image

if **name** == **main**:
learning() # Fashion Mnist Dataset 학습 # 이미지 예측 (이미지 경로를 적절히 수정하세요)
image_path = 'img/1.jpg'
predict_image(image_path) # 학습된 모델을 바탕으로 경로에 있는 이미지에서 패션 카테고리 추출
