import os
import cv2
import numpy as np
import pickle

def read_images_from_directory(dir_path):
    #image_names = ["HOG_descriptor.jpg", "Magnitude.jpg", "Orientation.jpg", "HOG_image2.jpg", "HOG_Feature_image.jpg"]
    image_names = ["HOG_descriptor.jpg", "HOG_Feature_image.jpg"]
    return [cv2.imread(os.path.join(dir_path, image_name), cv2.IMREAD_GRAYSCALE).flatten() for image_name in image_names]

def predict_label_for_directory(model, dir_path):
    features = read_images_from_directory(dir_path)
    concatenated_features = np.concatenate(features)
    return model.predict([concatenated_features])[0]

if __name__ == "__main__":
    model_path = "random_forest_model.pkl"
    
    # Load the trained model
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    # Specify the directory you want to test
    test_directory = r"C:\Users\nehal\Downloads\Compressed\drive-download-20230928T233402Z-001"

    # Iterate over the directories inside
    for label in os.listdir(test_directory):
        label_path = os.path.join(test_directory, label)
        if os.path.isdir(label_path):
            for sub_dir in os.listdir(label_path):
                full_path = os.path.join(label_path, sub_dir)
                if os.path.isdir(full_path):
                    predicted_label = predict_label_for_directory(clf, full_path)
                    print(f"True Label: {label} | Predicted Label: {predicted_label}")

