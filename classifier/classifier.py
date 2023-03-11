import torch
import torch.nn as nn
import time
import copy
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

# To train the CNN, we want to use transfer learning, which is when we use a CNN that has been trained for something else and we freeze all the weights except for the final fully connected layer
# Freezing all the other weights is helpful because since this CNN has already been trained, we can trust these weights are capable of extracting meaningful features.
# The last layer that we train, is the layer which can identify whether the face is ours or not

# Load Data
train_data = np.load(
    "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/train_data.npy",
    allow_pickle=True,
)
val_data = np.load(
    "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/val_data.npy",
    allow_pickle=True,
)


# Class that returns the images and corresponding labels from the data files
class Data(Dataset):
    def __init__(self, datafile):
        self.datafile = datafile

    def __len__(self):
        return self.datafile.shape[0]

    def __getitem__(self, idx):
        img = self.datafile[idx][0]  # img
        y = int(self.datafile[idx][1])  # label
        if y == 0:
            label = torch.tensor([1.0, 0.0])  # converting the int label to a tensor
        if y == 1:
            label = torch.tensor([0.0, 1.0])
        return img, label


train_data_class = Data(train_data)
val_data_class = Data(val_data)

# Create DataLoader objects
# Takes train and validation dataset objects and efficiently loads the data into batches during training and validation
# batch_size whihc indicates number of training examples processed in one iteration
# shuffle - if true data is shuffled at every epoch

train_loader = DataLoader(train_data_class, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data_class, batch_size=32, shuffle=True)

#

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": train_data.shape[0], "val": val_data.shape[0]}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build Model
# We are leveraging a preexisting model known as ResNet-18
# Consists of 18 layers which is used for image classification tasks

model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = (
        False  # all the parameters are frozen, we only want to train the last layer
    )

# Parameters of newly constructed modules have requires_grad=True

num_features = model_conv.fc.in_features  # num features of last layer
model_conv.fc = nn.Sequential(
    nn.Linear(num_features, 2), torch.nn.Sigmoid()
)  # create new layer


# Defining loss function and optimizer

criterion = (
    nn.MSELoss()
)  # Loss function is Mean Squared Error, measures difference between predicted and actual labels

optimizer_conv = optim.Adam(
    model_conv.fc.parameters(), lr=0.001
)  # Optimizer is Adam, an optimization algo that adjusts learning rate adaptively during training

# Defining learning rate scheduler
# Adjusts learning rate of optimizer during training
# StepLR scheduler is used here, multiplies learning rate by 'gamma' after every 'step_size' num of epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# # Testing with one batch

# image, label = next(iter(dataloaders["val"]))

# outputs = model_conv(image)
# value, preds = torch.max(outputs, 1)

# print(outputs)
# print(label)
# loss = criterion(outputs, label)


def train_model(model, optimizer, criterion, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # each epoch is one complete pass through the dataset during the training of a machine learning model.
    # NOTE: when it comes to epochs, its good to find a balance, too little epochs may
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate model

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # moves inputs tensor (img) to the device
                labels = labels.to(device)  # moves labels tensor to the device ("GPU")

                # zero parameter gradients
                optimizer.zero_grad()

                # Use both forward and backward propagation during training to update the model weights and minimize the training loss
                # only use forward propagation during evaluation to get the predictions of the model on the evaluation data and calculate the evaluation loss and accuracy.

                # forward learning
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    value, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # get statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == torch.argmax(labels, 1))

            if phase == " train":
                scheduler.step()

            # The best model weights are updated if the validation accuracy is better than the prev best accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            # deep copy model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")

    print(f"Best val Acc: {best_acc}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_conv = train_model(
    model_conv, optimizer_conv, criterion, exp_lr_scheduler, num_epochs=25
)

torch.save(
    model_conv.state_dict(),
    "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/model-resnet18.pth",
)
