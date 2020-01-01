from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 225.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

print("Want to load a model?")
load = input(">")

if load == 'y':
    model = keras.models.load_model("owo.h5")
else:
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(80, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    tester = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
predict = model.predict(test_images)

print("train model?")
train_me = input(">")
if train_me == "y":
    e = int(input("epochs> "))
    tester = model.fit(train_images, train_labels, epochs=e, validation_data=(test_images, test_labels))

print("save model?")
saver = input(">")
if saver == "y":
    model.save("owo.h5")
   
for owo in range(1000):
    plt.grid(False)
    plt.imshow(test_images[owo], cmap=plt.cm.binary)
    plt.xlabel("Guess: " + class_names[np.argmax(predict[owo])])
    plt.title(class_names[int(test_labels[owo])])
    plt.show()
