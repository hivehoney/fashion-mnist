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
