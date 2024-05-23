import os
import cv2
import numpy as np
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import cifar10

def preprocess_image(image):
    """
    Preprocess a single image by converting it to grayscale and normalizing the values.
    """
    return cv2.normalize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, 0, 1.0, cv2.NORM_MINMAX)

def preprocess_images(images):
    """
    Preprocess a batch of images by applying the preprocess_image function to each image.
    """
    return np.array([preprocess_image(image).flatten() for image in images])

def train_logistic_regression(X, y, random_state=42, max_iter=1000):
    """
    Train a logistic regression classifier on the preprocessed data.
    """
    clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
    clf.fit(X, y)
    return clf

def evaluate_model(clf, X, y, target_names, out_dir):
    """
    Evaluate the performance of the classifier and save the classification report to a file.
    """
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred, target_names=target_names)
    print(report)
    with open(os.path.join(out_dir, 'lrc_report.txt'), "w") as file:
        file.write(report)

def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = preprocess_images(X_train), preprocess_images(X_test)
    y_train, y_test = y_train.flatten(), y_test.flatten()
    img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    out_dir = 'out'
    
    clf = train_logistic_regression(X_train, y_train)
    dump(clf, os.path.join(out_dir, "lr_classifier.joblib"))
    evaluate_model(clf, X_test, y_test, img_labels, out_dir)

if __name__ == "__main__":
    main()