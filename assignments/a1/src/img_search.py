import os
import cv2
import argparse
import pandas as pd

def process_image(folder_path, image_to_compare):
    images = os.listdir(folder_path)
    dist = [("target", 0.0)]
    
    def comp_hist(image):
        comp_img1 = cv2.imread(os.path.join(folder_path, image_to_compare))
        comp_img2 = cv2.imread(os.path.join(folder_path, image))
        
        hist_1 = cv2.calcHist([comp_img1], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        hist_2 = cv2.calcHist([comp_img2], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        
        normalized_hist_1 = cv2.normalize(hist_1, hist_1, 0, 1.0, cv2.NORM_MINMAX)
        normalized_hist_2 = cv2.normalize(hist_2, hist_2, 0, 1.0, cv2.NORM_MINMAX)

        new_dist = round(cv2.compareHist(normalized_hist_1, normalized_hist_2, cv2.HISTCMP_CHISQR), 2)
        dist.append((image, new_dist))
    
    for image in images:
        if image != image_to_compare:
            comp_hist(image)
    
    dist.sort(key=lambda dist: dist[1])
    dist = dist[:6]
    df = pd.DataFrame(dist, columns=["Filename", "Distance"])
    df.to_csv("out/similar_images.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_to_compare', type=str, help='Image filename to compare')
    args = parser.parse_args()
    
    folder_path = os.path.join("data", "flowers")
    image_to_compare = args.image_to_compare
    
    process_image(folder_path, image_to_compare)
    print("Images processed successfully.")

if __name__ == "__main__":
    main()
