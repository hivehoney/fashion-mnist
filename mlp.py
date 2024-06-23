import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1)

def create_deep_model(neurons, input_shape, num_classes):
    model = Sequential([
        Dense(neurons, input_shape=(input_shape,), activation='relu'),
        Dropout(0.5),
        Dense(neurons, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def learn_model(train_images, train_classes, test_images, test_classes):
    # model 학습 향상 및 속도를 위한 정규화 (0~1)
    train_images = train_images.reshape((train_images.shape[0], 784)).astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 784)).astype('float32') / 255

    num_classes = 10
    train = to_categorical(train_classes, num_classes)
    test = to_categorical(test_classes, num_classes)

    # 뉴런 수를 다르게 설정하여 모델 학습
    neuron_options = [512]

    # Dictionary to store histories for plotting
    histories = {}

    for neurons in neuron_options:
        # 조기 종료 콜백
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 학습률 조정 콜백
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        print(f"\nTraining model with {neurons} neurons")
        model = create_deep_model(neurons, input_shape=784, num_classes=num_classes)

        filepath = "mlp-weights.{epoch:02d}-{val_loss:.2f}.keras"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

        history = model.fit(train_images, train,
                            batch_size=128,
                            epochs=10,
                            validation_data=(test_images, test),
                            callbacks=[early_stopping, lr_scheduler, checkpoint])

        # Save the history for plotting
        histories[neurons] = history.history

        # 모델 평가
        loss, accuracy = model.evaluate(test_images, test)
        print(f"Test accuracy with {neurons} neurons: {accuracy}")
        print(f"Test loss with {neurons} neurons: {loss}")

    # 학습된 모델 저장
    model.save('fashion_mnist_mlp.h5')

    # .keras 파일 삭제
    for file in os.listdir():
        if file.endswith(".keras"):
            os.remove(file)

    return histories, neuron_options, test_images, test_classes

def plot_results(histories, neuron_options):
    # Plot the training and validation loss for each configuration of neurons
    plt.figure(figsize=(15, 10))

    for i, neurons in enumerate(neuron_options):
        plt.subplot(2, 2, i + 1)
        plt.plot(histories[neurons]['loss'], label='Train Loss')
        plt.plot(histories[neurons]['val_loss'], label='Validation Loss')
        plt.title(f'Loss Curve for {neurons} Neurons')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    plt.tight_layout()
    plt.savefig('img/test/training_validation_loss.png')  # Save the plot as an image
    plt.close()  # Close the plot to free up memory

def plot_predictions(model, test_images, test_classes):
    # Define class names
    CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Function to plot an image
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i].reshape(28, 28)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(CLASSES[predicted_label],
                                             100 * np.max(predictions_array),
                                             CLASSES[true_label]),
                   color=color)

    # Function to plot the value array
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    # Example usage
    # Assuming predictions is a variable that contains the model's predictions for the test set
    predictions = model.predict(test_images.reshape(-1, 784))

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_classes, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_classes)
    plt.savefig('img/test/predictions.png')  # Save the plot as an image
    plt.close()  # Close the plot to free up memory

def learning():
    # Set random seeds for reproducibility
    np.random.seed(1)
    tf.random.set_seed(1)

    # 카테고리 분류
    CLASSES = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

    # fashion mnist setting
    # image size: 28x28
    (train_images, train_classes), (test_images, test_classes) = fashion_mnist.load_data()

    # 데이터를 학습용과 검증용으로 분할
    train_images, val_images, train_classes, val_classes = train_test_split(train_images, train_classes, test_size=0.2, random_state=42)

    # 데이터셋의 크기 출력
    print(f'훈련 데이터 이미지 개수: {train_images.shape[0]}')
    print(f'훈련 데이터 레이블 개수: {train_classes.shape[0]}')
    print(f'검증 데이터 이미지 개수: {val_images.shape[0]}')
    print(f'검증 데이터 레이블 개수: {val_classes.shape[0]}')
    print(f'테스트 데이터 이미지 개수: {test_images.shape[0]}')
    print(f'테스트 데이터 레이블 개수: {test_classes.shape[0]}')

    # 모델 학습
    histories, neuron_options, test_images, test_classes = learn_model(train_images, train_classes, test_images, test_classes)

    # 학습된 모델 로드
    model = load_model('fashion_mnist_mlp.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    plot_results(histories, neuron_options)
    plot_predictions(model, test_images, test_classes)