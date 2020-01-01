import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=12)

prediction = model.predict(test_images)

for owo in range(20):
    plt.grid(False)
    plt.imshow(test_images[owo], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[owo]])
    plt.title(class_names[np.argmax(prediction[owo])])
    plt.show()
print(class_names[np.argmax(prediction[0])])
