import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import randrange

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.load_model("hand_written_digits_rec.h5")

# visualize all predictions
while True:
    index = randrange(0, 10000)
    prediction = model.predict([x_test])
    print("Prediction:", np.argmax(prediction[index]))
    print("Actual digit:", y_test[index])

    plt.imshow(x_test[index], cmap=plt.cm.binary)
    plt.show()

# # visialize wrong predictions only
# while True:
#     index = randrange(0, 10000)
#     prediction = model.predict([x_test])
#     prediction = np.argmax(prediction[index])
#     ac_digit = y_test[index]
#     if prediction == ac_digit: continue
#     print("Prediction:", prediction)
#     print("Actual digit:", ac_digit)

#     plt.imshow(x_test[index], cmap=plt.cm.binary)
#     plt.show()