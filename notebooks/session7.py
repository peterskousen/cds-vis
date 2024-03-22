import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
data = data.astype("float")/255.0

(X_train, X_test, y_train, y_test) = train_test_split(data,
                                                    labels, 
                                                    test_size=0.2)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

model = Sequential() # Sequential = goes through each layer one by one
model.add(Dense(256, #Add layer of 256 fully-connected nodes
                input_shape=(784,), 
                activation="relu"))
model.add(Dense(128, 
                activation="relu"))
model.add(Dense(10, 
                activation="softmax"))

model.summary()

plot_model(model, show_shapes=True, show_layer_names=True)

sgd = SGD(learning_rate = 0.01)
model.compile(loss="categorical_crossentropy", # Name of the loss function in use; there are many different 
              optimizer=sgd, 
              metrics=["accuracy"])

history = model.fit(X_train, y_train, 
                    validation_split=0.1, # Keep 10% of the training data for continously validating remaining 90% of data
                    epochs=10, # Going through entire dataset 10 times, adjusting values as it goes
                    batch_size=32) 


# ## Visualise using ```matplotlib```
def plot_results():
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, 10), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 10), history.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(np.arange(0, 10), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 10), history.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

# ## Classifier metrics

# evaluate network
print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=32)

print(classification_report(y_test.argmax(axis=1), 
                            predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_]))


def mainr():
    parser = argparse.ArgumentParser(description = "Loading & printing array")
    # parser.add_argument("--input", 
    #                     "-i",
    #                     required=True,
    #                     help="Filepath to CSV to load and print")
    args = parser.parse_args()
    return args