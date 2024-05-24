import os
import cv2
import pandas as pd
from tqdm import tqdm

def process_image(folder_path, image_to_compare):
    images = os.listdir(folder_path)

    # Initialize a list to store distances between images
    dist = [("target", 0.0)]
    
    def comp_hist(image):
        comp_img1 = cv2.imread(os.path.join(folder_path, image_to_compare))
        comp_img2 = cv2.imread(os.path.join(folder_path, image))
        
        # Compute histograms for both images
        hist_1 = cv2.calcHist([comp_img1], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        hist_2 = cv2.calcHist([comp_img2], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        
        # Normalize the histograms
        normalized_hist_1 = cv2.normalize(hist_1, hist_1, 0, 1.0, cv2.NORM_MINMAX)
        normalized_hist_2 = cv2.normalize(hist_2, hist_2, 0, 1.0, cv2.NORM_MINMAX)

        # Calculate the chi-square distance between the histograms
        new_dist = round(cv2.compareHist(normalized_hist_1, normalized_hist_2, cv2.HISTCMP_CHISQR), 2)
        
        # Append the computed distance to the list
        dist.append((image, new_dist))
    
    print("Finding similar images...")
    for image in tqdm(images):
        if image != image_to_compare:
            comp_hist(image)
    
    dist.sort(key=lambda dist: dist[1])
    dist = dist[:6]
    df = pd.DataFrame(dist, columns=["Filename", "Distance"])
    df.to_csv("out/similar_images.csv", index=False)
    print("Image search completed. Results saved to out folder")

if __name__ == "__main__":
    image_to_compare = input(f'Enter the filename # of the image to compare: image_')
    image_to_compare = 'image_' + image_to_compare + '.jpg'
    input_path = os.path.join("in", "flowers")
    process_image(input_path, image_to_compare)