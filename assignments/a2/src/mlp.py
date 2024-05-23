import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import cifar10

def preprocess_image(image):
    """
    Preprocess a single image by converting it to grayscale and normalizing the values.
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    normalized_image = cv2.normalize(gray_image, None, 0, 1.0, cv2.NORM_MINMAX)
    return normalized_image

def preprocess_images(images):
    """
    Preprocess images
    """
    preprocessed_images = [preprocess_image(image).flatten() for image in images]
    return np.array(preprocessed_images)

def train_mlp_classifier(X, y, 
                        hidden_layer_sizes=(128,), 
                        max_iter=500, 
                        random_state=42, 
                        early_stopping=True, 
                        activation='relu'):
    """
    Train an MLP classifier on the preprocessed data.
    """

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                        max_iter=max_iter, 
                        random_state=random_state, 
                        early_stopping=early_stopping, 
                        activation=activation)

    clf.fit(X, y)
    return clf

def evaluate_model(clf, X, y, target_names, out_dir):
    """
    Evaluate the performance of the classifier, save the classification report to a file, and plot the loss curve.
    """

    y_pred = clf.predict(X)

    report = classification_report(y, y_pred, target_names=target_names)
    print(report)

    with open(os.path.join(out_dir,'mlp_report.txt'), "w") as file:
        file.write(report)

    plt.plot(clf.loss_curve_, label='Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('MLP Classifier Loss Curve during Training')
    plt.legend()

    plt.savefig(os.path.join(out_dir, 'plot.png'))

    plt.show()

def main():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    out_dir = 'out'
    
    clf = train_mlp_classifier(X_train, y_train)
    dump(clf, os.path.join(out_dir, "mlp_classifier.joblib"))
    evaluate_model(clf, X_test, y_test, img_labels, out_dir)

if __name__ == "__main__":
    main()