# predict.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 모델 로드
model_path = 'fashion_mnist_mlp.h5'
model = load_model(model_path)

# Fashion MNIST 클래스 라벨
class_labels_dict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

# 이미지 전처리 함수 (색상 반전 포함)
def preprocess_image(img_path):
    # Load image and convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {img_path}")

    # Resize image
    img_resized = cv2.resize(img, (28, 28))

    # Invert colors
    img_inverted = cv2.bitwise_not(img_resized)

    # Enhance the contrast to ensure the object is clearly visible
    alpha = 3.0  # Simple contrast control
    beta = 5  # Simple brightness control
    img_contrast = cv2.convertScaleAbs(img_inverted, alpha=alpha, beta=beta)

    # Convert the image to fit the model input format
    img_processed = img_contrast.reshape(1, 784).astype('float32') / 255

    return img_processed

# 전체 이미지 예측 함수
def predict_full_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_labels_dict[predicted_class[0]]
    return predicted_label, predictions

# 이미지 로드 및 전처리
def predict_image(image_path):
    processed_img = preprocess_image(image_path)

    # 전체 이미지 예측
    predicted_label, predictions = predict_full_image(model, processed_img)

    # 전처리된 img_array 시각화
    plt.figure(figsize=(5, 5))
    plt.imshow(processed_img.reshape(28, 28), cmap='gray')
    plt.title("Processed Image (img_array)")
    plt.savefig('img/result/processed_image.png')
    plt.close()

    # 결과 출력
    print(f"The predicted class for the provided image is: {predicted_label}")

    # 각 클래스별 확률 출력
    for i, prob in enumerate(predictions[0]):
        print(f"{class_labels_dict[i]}: {prob:.4f}")

    # 원본 이미지 시각화
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    original_img = cv2.bitwise_not(original_img)  # 필요시 색상 반전
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Provided Image\nPredicted class: {predicted_label}")

    plt.subplot(1, 2, 2)
    bars = plt.bar(class_labels_dict.values(), predictions[0])
    plt.title("Class Probabilities")
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    for bar, prob in zip(bars, predictions[0]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{prob:.4f}', ha='center', va='bottom')

    plt.savefig('img/result/class_probabilities.png')
    plt.close()
