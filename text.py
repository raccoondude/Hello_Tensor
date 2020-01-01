import tensorflow as tf
from tensorflow import keras
import numpy as np

#Data here
print("edit default values")
values = input("[y/n] >")

if values == "y":
    epoch = int(input("epoch >"))
    NN = int(input("NN >"))
else:
    epoch = 50
    NN = 60


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_me(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

def encode_me(text):
    encoded = [1]
    for word in text:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

print("make and train new model?")
hello = input("[y/n] >")
if hello == "y":

    model = keras.Sequential()
    model.add(keras.layers.Embedding(88000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(NN, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fitModel = model.fit(x_train, y_train, epochs=epoch, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)

else:
    model = keras.models.load_model("owo.h5")

print("check data in dataset")
check_data = input("[y/n] >")
if check_data == "y":
    while True:
        print("Select number")
        uwu = input(">")
        if uwu == "quit":
            break;
        test_review = test_data[int(uwu)]
        predict = model.predict([test_review])
        print("review:")
        print(decode_me(test_review))
        print("Predicition: " + str(predict[0]))
        print("Actal: " + str(test_labels[0]))

print("check model with external data?")
check_external = input("[y/n] >")
if check_external == "y":
    file_name = input("file name>")
    with open("uwu", encoding="utf-8") as f:
        for line in f.readlines():
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").split(" ")
            encode = encode_me(nline)
            encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
            predict = model.predict(encode)
            print(line)
            print(encode)
            print(predict[0])
if hello == "y":
    print("save the model?")
    saving = input("[y/n] >")
    if saving == "y":
        model.save("owo.h5")
