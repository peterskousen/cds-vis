import os
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors

def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=False)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)
    return normalized_features

def load_model_and_images(target_image):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    target_image = os.path.join("in", "flowers", target_image)
    root_dir = os.path.join("in", "flowers")
    filenames = [root_dir + "/" + name for name in sorted(os.listdir(root_dir))]
    return model, target_image, filenames

def find_nearest_neighbors(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return distances, indices

def plot_images(filenames, idxs, distances, save_path):
    plt.imshow(mpimg.imread(filenames[250]))
    plt.title("Target Image")
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.suptitle('Top 3 Nearest Neighbors with Distances')
    for i, idx in enumerate(idxs, 1):
        plt.subplot(3, 3, i)
        plt.imshow(mpimg.imread(filenames[idx]))
        plt.title(f"Top {i}\nDistance: {distances[0][i-1]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'nearest_neighbors_with_distances.png'))
    plt.close()

def main(target_image, save_path):
    model, target_image, filenames = load_model_and_images(target_image)
    features = extract_features(target_image, model)
    
    feature_list = [extract_features(filename, model) for filename in tqdm(filenames)]
    
    distances, indices = find_nearest_neighbors(features, feature_list)
    
    idxs = [indices[0][i] for i in range(1, 6)]
    
    plot_images(filenames, idxs, distances, save_path)

if __name__ == "__main__":
    target_image = input(f'Enter the filename # of the image to compare: image_')
    target_image = 'image_' + target_image + '.jpg'
    save_path = "out"
    os.makedirs(save_path, exist_ok=True)
    main(target_image, save_path)
