import cv2
import numpy as np
import os

def compute_hog_descriptor(magnitude, orientation):
    cell_size = 8
    block_size = 2
    nbins = 9
    bin_width = 180.0 / nbins
    histogram = np.zeros((magnitude.shape[0] // cell_size, magnitude.shape[1] // cell_size, nbins))

    # Orientation binning
    for i in range(0, magnitude.shape[0] - cell_size, cell_size):
        for j in range(0, magnitude.shape[1] - cell_size, cell_size):
            cell_magnitude = magnitude[i:i+cell_size, j:j+cell_size]
            cell_orientation = orientation[i:i+cell_size, j:j+cell_size]
            cell_histogram = np.zeros(nbins)

            for k in range(cell_size):
                for l in range(cell_size):
                    bin_idx = int(cell_orientation[k, l] // bin_width)
                    bin_idx = np.clip(bin_idx, 0, nbins-1)
                    cell_histogram[bin_idx] += cell_magnitude[k, l]

            histogram[i//cell_size, j//cell_size, :] = cell_histogram

    # Descriptor blocks
    hog_image = np.zeros_like(magnitude, dtype=np.uint8)
    for i in range(0, histogram.shape[0] - block_size + 1):
        for j in range(0, histogram.shape[1] - block_size + 1):
            block_histogram = histogram[i:i+block_size, j:j+block_size].ravel()
            dominant_gradient = np.argmax(block_histogram)
            center_x, center_y = j*cell_size*block_size + cell_size, i*cell_size*block_size + cell_size
            delta_x, delta_y = cell_size * np.cos(dominant_gradient * bin_width * np.pi / 180.0), cell_size * np.sin(dominant_gradient * bin_width * np.pi / 180.0)
            cv2.arrowedLine(hog_image, (int(center_x - delta_x/2), int(center_y - delta_y/2)), (int(center_x + delta_x/2), int(center_y + delta_y/2)), 255, 1, tipLength=0.3)

    return hog_image

base_directory = r"C:\Users\nehal\Downloads\Compressed\drive-download-20230928T233402Z-001"

for root, dirs, files in os.walk(base_directory):
    if "Magnitude.jpg" in files and "Orientation.jpg" in files:
        magnitude_path = os.path.join(root, "Magnitude.jpg")
        orientation_path = os.path.join(root, "Orientation.jpg")
        hog_save_path = os.path.join(root, "HOG_image.jpg")

        print(f"Processing from magnitude and orientation images in: {root}")

        try:
            magnitude = cv2.imread(magnitude_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            orientation = cv2.imread(orientation_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

            hog_image = compute_hog_descriptor(magnitude, orientation)
            cv2.imwrite(hog_save_path, hog_image)

        except Exception as e:
            print(f"An error occurred while processing images in {root}: {e}")
