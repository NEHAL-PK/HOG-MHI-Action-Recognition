import cv2
import os
import numpy as np

def compute_MHI(image_folder, tau):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
    if not image_files:
        raise ValueError("No images found in the specified directory!")
    
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first_image.shape
    MHI = np.zeros((h, w), dtype=np.uint8)
    
    decay_value = tau / len(image_files)

    for i in range(1, len(image_files)):
        prev_image = cv2.imread(os.path.join(image_folder, image_files[i-1]), cv2.IMREAD_GRAYSCALE)
        curr_image = cv2.imread(os.path.join(image_folder, image_files[i]), cv2.IMREAD_GRAYSCALE)
        
        # Compute difference between the current and the previous frame
        diff = cv2.absdiff(curr_image, prev_image)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)  # Lower threshold for finer motion detection
        
        # Update MHI
        MHI = np.where(motion_mask == 255, tau, np.maximum(MHI - decay_value, 0))

    return MHI

base_directory = r"C:\Users\nehal\Downloads\Compressed\data\New folder (2)"
tau = 255  # max value

# Iterate through every directory inside the base directory
for root, dirs, files in os.walk(base_directory):
    if files and any(file.endswith('.jpg') for file in files):  # Check if there are jpg files in the directory
        MHI_image = compute_MHI(root, tau)
        
        # Save the MHI image
        save_path = os.path.join(root, "MHI_result.jpg")
        cv2.imwrite(save_path, MHI_image)

        # Display the MHI image (optional)
        cv2.imshow('MHI', MHI_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
