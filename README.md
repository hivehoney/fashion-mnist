# fashion-mnist

## Fashion Mnist Data를 학습시켜서 이미지에서 패션 카테고리 추출하기

### 코드 실행 전 모듈 설치

1. pip install numpy
2. pip install opencv-python
3. pip install tensorflow
4. pip install matplotlib
5. pip install scikit-learn
6. pip install os-sys
7. pip install keras

### 코드 수행 절차

1. img 폴더에 사진 파일 업로드
2. main.py 실행
3. mlp.py에서 learning 함수를 호출하여 fashion mnist data set을 불러와서 학습시킨다.
4. data set 학습 완료 후 predict.py에서 predict_image 함수를 호출하여 학습된 모델을 바탕으로 경로에 있는 이미지에서 패션 카테고리 추출.

### main.py

from mlp import learning
from predict import predict_image

if **name** == **main**:
learning() # Fashion Mnist Dataset 학습 # 이미지 예측 (이미지 경로를 적절히 수정하세요)
image_path = 'img/1.jpg'
predict_image(image_path) # 학습된 모델을 바탕으로 경로에 있는 이미지에서 패션 카테고리 추출
