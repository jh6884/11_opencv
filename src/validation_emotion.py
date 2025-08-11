import cv2, dlib
import matplotlib.pyplot as plt
import tensorboard as tf
import numpy as np

image = cv2.imread('../img/charles-etoroma.jpg')

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# cv2.imshow('charles', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

face_detector = dlib.cnn_face_detection_model_v1('../models/mmod_human_face_detector.dat')

face_detection = face_detector(face_detector, 1)

network = tf.keras.models.load.model('../img/models/emotion_model.h5')

left, top, right, bottom = face_detection[0].rect.left(), face_detection[0].rect.top(), face_detection[0].rect.right(), face_detection[0].rect.bottom()

pred_probability = network.medols.image
roi = image[top:bottom, left:right]

cv2.imshow(roi)

print(roi.shape)

# normalize
roi = roi / 255

roi = np.expand_dims(roi, axis=0)