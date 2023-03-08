import os
import numpy as np

from torchvision import transforms
from torchvision import datasets, transforms

# Data transforms applied to images, resizing to 224x224px, converting to tensor, then normalizing it.

# Why normalize it?
# Important preprocessing step, helps adjust model to learn efficiently and effectively.

data_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_dir = "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/dataset"

# lambda fn -> loads data from the data_directory, applies corresponding transformation based on the type of data it is (but because theyre both images in the same format, transformation is the same)
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform)
    for x in ["train", "validation"]
}

# Save as npy files -> npy files store all info required to construct an array
# required as this is the data that the classifier will take in and use to train the layer
train_data = []

count = 0
for img, label in image_datasets["train"]:
    train_data.append([np.array(img), np.array(label)])

train_data_np = np.array(train_data, dtype=object)  # why does this work????
np.save(
    "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/"
    + "train_data"
    + ".npy",
    train_data_np,
)


val_data = []

for img, label in image_datasets["validation"]:
    val_data.append([np.array(img), np.array(label)])

val_data_np = np.array(val_data, dtype=object)
print(val_data)
np.save(
    "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/"
    + "val_data"
    + ".npy",
    val_data_np,
)
