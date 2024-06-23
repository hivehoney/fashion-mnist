from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import optuna


# 모델 생성 함수
def create_model(trial):
    model = Sequential()
    model.add(Dense(trial.suggest_int('units_1', 128, 512), input_shape=(784,), activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout_1', 0.2, 0.5)))
    model.add(Dense(trial.suggest_int('units_2', 128, 512), activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout_2', 0.2, 0.5)))
    model.add(Dense(num_classes, activation='softmax'))

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Objective function
def objective(trial):
    model = create_model(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(train_images, train,
                        batch_size=trial.suggest_int('batch_size', 64, 256),
                        epochs=50,
                        validation_data=(test_images, test),
                        callbacks=[early_stopping, lr_scheduler],
                        verbose=0)

    loss, accuracy = model.evaluate(test_images, test, verbose=0)
    return accuracy

# Optuna study 생성 및 하이퍼파라미터 튜닝 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 최적 하이퍼파라미터 출력
print("Best hyperparameters: ", study.best_params)

# 최적 하이퍼파라미터로 최종 모델 학습
best_params = study.best_params
final_model = create_model(optuna.trial.FixedTrial(best_params))
final_model.fit(train_images, train,
                batch_size=best_params['batch_size'],
                epochs=50,
                validation_data=(test_images, test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                           ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)],
                verbose=1)

# 최종 모델 평가
loss, accuracy = final_model.evaluate(test_images, test)
print(f"Final test accuracy: {accuracy}")
print(f"Final test loss: {loss}")




import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 예측 함수
def predict_image(img_array):
    # 예측
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    # 결과 해석
    predicted_label = class_labels_dict[predicted_class[0]]
    return predicted_label

# 이미지 로드 및 전처리
image_path = '/content/img1.png'  # 제공된 이미지 파일 경로
original_img = Image.open(image_path)
original_img = original_img.convert('L')  # 흑백으로 변환

# 이미지 리사이즈 (원본 비율 유지)
resize_factor = 0.18  # 원하는 크기에 맞게 조정
new_size = (int(original_img.width * resize_factor), int(original_img.height * resize_factor))
img = original_img.resize(new_size, Image.ANTIALIAS)


# 이미지 크기
width, height = img.size

# 슬라이딩 윈도우 설정
window_size = 56  # 윈도우 크기 (28x28의 두 배로 설정)
step_size = 20    # 슬라이딩 스텝 (오버랩을 줄이기 위해 절반으로 설정)

# 결과 저장
bounding_boxes = []
predicted_labels = []

# 슬라이딩 윈도우를 사용하여 이미지 분할 및 예측
for top in range(0, height - window_size + 1, step_size):
    for left in range(0, width - window_size + 1, step_size):
        right = left + window_size
        bottom = top + window_size
        img_section = img.crop((left, top, right, bottom))

        # 이미지를 모델 입력 형식에 맞게 전처리
        img_section = img_section.resize((28, 28))
        img_array = np.array(img_section).reshape((1, 784)).astype('float32') / 255

        # 예측
        predicted_label = predict_image(img_array)
        predicted_labels.append(predicted_label)
        bounding_boxes.append((left, top, right, bottom))

# 결과 시각화
image = cv2.imread(image_path)
image = cv2.resize(image, new_size)
for i, (left, top, right, bottom) in enumerate(bounding_boxes):
    label = predicted_labels[i]
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Sliding Window Detection')
plt.axis('off')
plt.show()

# 검출된 객체의 카테고리 출력
unique_predicted_labels = list(set(predicted_labels))
for i, label in enumerate(unique_predicted_labels):
    print(f"Predicted Label {i+1}: {label}")