from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class LeNet:
    @staticmethod
    def build(width, height, channels, classes):
        # initialize the model
        input_shape = (height, width, channels)
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)

        model = Sequential()
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax", name="output"))

        return model

def main():

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    print("[INFO] accessing MNIST...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
    	X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    	X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
    else:
    	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # scale data to the range of [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # convert the labels from integers to vectors
    le = LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    optimizer = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, channels=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(
        X_train, y_train,
    	validation_data=(X_test, y_test),
        batch_size=128,
    	epochs=20,
        verbose=1
    )

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=128)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

    # Save model
    # model.save("mnist.model")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 20), H.history["loss"], label="train loss")
    plt.plot(np.arange(0, 20), H.history["val_loss"], label="validation loss")
    plt.plot(np.arange(0, 20), H.history["accuracy"], label="train accuracy")
    plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="validation accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
plt.show()
