import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.metrics import accuracy_score


img = tf.keras.preprocessing.image.load_img('../data/train/happy/Training_1206.jpg')

print(np.array(img).shape)

# make train, test dataset
# costruct cnn model with tensorflow

train_generator = ImageDataGenerator(rotation_range=10,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     rescale=1/255)

train_dataset = train_generator.flow_from_directory(directory='../data/train',
                                                    target_size=(48,48),
                                                    class_mode='categorical',
                                                    batch_size=16,
                                                    shuffle=True,
                                                    seed=10)

test_generator = ImageDataGenerator(rescale=1/255)

test_dataset = test_generator.flow_from_directory(directory='/content/fer2013/validation',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)

# 훈련 데이터셋의 타깃값
#print(train_dataset.classes)

# 각 타깃값의 의미
#print(train_dataset.class_indices)

# 타깃값별로 개수 세기
#print(np.unique(train_dataset.classes, return_counts=True))

num_classes = 7
num_detectors = 32
width, height = 48, 48



network = Sequential()

network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same', input_shape=(width, height, 3)))
network.add(BatchNormalization())
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Flatten())

network.add(Dense(2*2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(num_classes, activation='softmax'))

network.summary()

# 모델 훈련
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 5

network.fit(train_dataset, epochs=epochs)

# 모델 성능 평가
network.evaluate(test_dataset)
preds = network.predict(test_dataset)
print(preds)
preds = np.argmax(preds, axis=1)
print(preds)
print(test_dataset.classes)
print(accuracy_score(test_dataset.classes, preds))