import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

img = tf.keras.processing.img.load_img('../../data/train/happy/Training_1206.jpg')

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()