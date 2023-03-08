import os
import torch
import torch.nn as nn

from facenet_pytorch import MTCNN
from face_detector import FaceDetector
from torchvision import models

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_features, 2), torch.nn.Sigmoid())

model.load_state_dict(
    torch.load(
        "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/model-resnet18.pth"
    )
)
model.eval()

# What is MTCNN?
# Multi-task Cascaded Convolutional Networks (MTCNN) is a framework developed as a solution for both face detection and face alignment. The process consists of three stages of convolutional networks that are able to recognize faces and landmark location such as eyes, nose, and mouth.

# MTCNN is a way to integrate both tasks (recognition and alignment) using multi-task learning. In the first stage it uses a shallow CNN to quickly produce candidate windows. In the second stage it refines the proposed candidate windows through a more complex CNN. And lastly, in the third stage it uses a third CNN, more complex than the others, to further refine the result and output facial landmark positions.

mtcnn = MTCNN()
fcd = FaceDetector.FaceDetector(mtcnn, classifier=model)
fcd.run()
