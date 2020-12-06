import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2

model = tf.keras.models.load_model("hand_written_digits_rec.h5")

path = "C:/Users/iya-w/Desktop/python/Tensorflow/hand_writen_digits_rec/my_own_hand_writen_digits"

numbers = []

for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) / 255
    numbers.append(img_array)

numbers = np.array(numbers)

for i in range(20):
    prediction = model.predict([numbers])
    print("Prediction:", np.argmax(prediction[i]))

    plt.imshow(numbers[i], cmap=plt.cm.binary)
    plt.show()