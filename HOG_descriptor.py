import cv2
import numpy as np
import os

def concatenate_descriptors_from_images(image_paths):
    descriptors = []
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        descriptors.append(image.ravel())  # Flatten the image
    concatenated_descriptor = np.hstack(descriptors)  # Stack them horizontally
    return concatenated_descriptor

base_directory = r"C:\Users\nehal\Downloads\Compressed\drive-download-20230928T233402Z-001"

for root, dirs, files in os.walk(base_directory):
    if all(x in files for x in ["HOG_descriptor.jpg", "Magnitude.jpg", "Orientation.jpg", "HOG_image2.jpg", "HOG_Feature_image.jpg"]):
        image_paths = [os.path.join(root, fname) for fname in ["HOG_descriptor.jpg", "Magnitude.jpg", "Orientation.jpg", "HOG_image2.jpg", "HOG_Feature_image.jpg"]]
        
        descriptor_save_path_npy = os.path.join(root, "Final_HOG_descriptor.npy")
        descriptor_save_path_csv = os.path.join(root, "Final_HOG_descriptor.csv")
        descriptor_save_path_img = os.path.join(root, "Final_HOG_descriptor.jpg")
        
        try:
            concatenated_descriptor = concatenate_descriptors_from_images(image_paths)
            
            # Reshape the descriptor to form an image (it'll be a long horizontal line)
            descriptor_image = concatenated_descriptor.reshape(1, -1)
            
            # Normalize for visualization
            descriptor_image_normalized = ((descriptor_image - descriptor_image.min()) * (255.0 / (descriptor_image.max() - descriptor_image.min()))).astype(np.uint8)
            
            # Save the descriptor as .npy
            np.save(descriptor_save_path_npy, concatenated_descriptor)
            
            # Save the descriptor as .csv
            np.savetxt(descriptor_save_path_csv, [concatenated_descriptor], delimiter=",")
            
            # Save the descriptor as an image
            cv2.imwrite(descriptor_save_path_img, descriptor_image_normalized)
            
        except Exception as e:
            print(f"An error occurred while processing images in {root}: {e}")
