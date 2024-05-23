import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

def process_images(path_in):
    data = []
    labels = []
    i = 0

    for folder in sorted(os.listdir(path_in)):
        label_path = os.path.join(path_in, folder)
        print(f'Processing images in "{label_path}"...')

        for img in os.listdir(label_path):
            if not img.endswith('.db'):
                image = load_img(os.path.join(label_path, img), target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                data.append(image)
                labels.append(i)
            else:
                continue
        i += 1

    return np.array(data), np.array(labels)

def encocde_labels(path_in, y_train_, y_test_):
    '''
    create one-hot encodings
    '''
    lb = LabelBinarizer()

    y_train = lb.fit_transform(y_train_)
    y_test = lb.fit_transform(y_test_)

    labelNames = []
    for label in sorted(os.listdir(path_in)):
        labelNames.append(label)

    return y_train, y_test, labelNames

def init_model():
    '''
    Initializes VGG16 model with custom classifier layers
    '''
    tf.keras.backend.clear_session()
    # Loads the VGG16 model without the fully connected top layers, 
    # sets the input shape to match the processed images = 224 X 224 X 3
    model = VGG16(include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3))

    # Marks loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # Adds new classifier layers on top of the pre-trained VGG16 model
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(10, activation='softmax')(class1)

    # Defines new model with the added layers 
    model = Model(inputs=model.inputs, 
                outputs=output)
    print("Compiling model...")

    # compile model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)

    sgd = SGD(
        learning_rate=lr_schedule)

    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    return model

def fit_model(initialized_model, X_train_, y_train_):
    H = initialized_model.fit(X_train_, y_train_, 
                validation_split=0.1,
                batch_size=128,
                epochs=10,
                verbose=1)

    return H

def eval_model(model, X_test_, y_test_,labelNames):
    print("Evaluating model performance")

    predictions = model.predict(X_test_, batch_size=128)

    report = classification_report(y_test_.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames)

    return report

def plot_history(H, epochs, out_path):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plt.show(block=True)
    plt.savefig(out_path)

def write_report(evaluation_report, path_):
    with open(path_, "w") as file:
        file.write(evaluation_report)

def main():
    path_in = os.path.join("in", "Tobacco3482")
    path_out_report = os.path.join("out", "classification_report.txt")
    path_out_plot = os.path.join("out", "plots.png")
    data, labels = process_images(path_in)

    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                          labels, 
                                                          test_size=0.2)
    y_train, y_test, labelNames = encocde_labels(path_in, y_train, y_test)

    print("Initializing model...")
    model = init_model()
    print("Training model, this might take a while...")

    H = fit_model(model, X_train, y_train)
    plot_history(H, 10, path_out_plot)
    evaluation_report = eval_model(model, X_test, y_test, labelNames)
    write_report(evaluation_report, path_out_report)

if __name__ == "__main__":
    main()