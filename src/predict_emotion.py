import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# 훈련 데이터셋의 타깃값
print(train_dataset.classes)

# 각 타깃값의 의미
print(train_dataset.class_indices)

# 타깃값별로 개수 세기
print(np.unique(train_dataset.classes, return_counts=True))