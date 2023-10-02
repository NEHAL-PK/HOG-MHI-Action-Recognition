
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut
import pickle
import matplotlib.pyplot as plt
import itertools

path = r"C:\Users\nehal\Downloads\Compressed\drive-download-20230928T233402Z-001"
labels = ["GuardToPunch", "KickRight", "PunchRight", "RunLeftToRight", "RunRightToLeft", "StandupLeft",
          "StandupRight", "TurnBackLeft", "TurnBackRight", "WalkLeftToRight", "WalkRightToLeft"]

def read_images_from_directory(dir_path):
    image_names = ["HOG_descriptor.jpg","HOG_Feature_image.jpg"]
    return [cv2.imread(os.path.join(dir_path, image_name), cv2.IMREAD_GRAYSCALE).flatten() for image_name in image_names]

X = []
y = []

# Prepare the data
for label in labels:
    label_path = os.path.join(path, label)
    subdirs = [d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d))]
    
    for sub in subdirs:
        features = read_images_from_directory(os.path.join(label_path, sub))
        concatenated_features = np.concatenate(features)
        X.append(concatenated_features)
        y.append(label)

X = np.array(X)
y = np.array(y)

loo = LeaveOneOut()
clf = RandomForestClassifier()

all_true_labels = []
all_pred_labels = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    all_true_labels.extend(y_test)
    all_pred_labels.extend(y_pred)

# Overall Classification Report
print(classification_report(all_true_labels, all_pred_labels))

# Overall Accuracy
overall_accuracy = accuracy_score(all_true_labels, all_pred_labels)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

# Displaying the confusion matrix
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
plt.figure(figsize=(15, 15))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > thresh else "black"
    if i != j:
        color = "red"
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color=color)

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save the model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(clf, f)
