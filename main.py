from mlp import learning
from predict import predict_image

if __name__ == "__main__":
    learning() # Fashion Mnist Dataset 학습
    # 이미지 예측 (이미지 경로를 적절히 수정하세요)
    image_path = 'img/1.jpg'
    predict_image(image_path)