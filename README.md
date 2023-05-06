# Facial Recognitition App

A Computer Vision project that identifies a person's face and blurs it. Used OpenCV, Pytorch, and Python.

This project leverages pretrained model alongside Transfer Learning to give it the functionality of identifying a particular person's face.

Code is heavily commented from my own learning as I started off completely new to the computer vision and deep learning space.

At 15 Epochs, model garnered an accuracy of ~99% accuracy

## Setup

The file structure is already setup. You will need to add in the proper files, folder, and paths.

You will need:
- A videos of yourself (short one, ~20-30 seconds)
- A folder of images of people that are not you (name it something like "not_me_unclean") 

One of the videos will be used for training and the other for validation.

The scripts `classifier/dataset/create_dataset_m.py`cuts up the video into frames that we can use to train the model (make sure the video includes different angles of your face). 

We will run this script that will put all of the frames somewhere then split the images in `classifier/dataset/train/me`, and `classifier/dataset/validation/me` (put majority in the training directory).

For the folder of images, I used a human face dataset on kaggle: https://www.kaggle.com/datasets/ashwingupta3012/human-faces

Download the files and use `create_dataset_nm.py` to get the images of the faces frames from the pictures. Split the images between `classifier/dataset/train/not_me` and `classifier/dataset/validation/not_me` (majority should be in the train directory).

From there, run `/classifier/prep_data.py`, this will transform all the images into a consistent format for better results in the training process.

Finally, run `/classifier/classifier.py`, which will perform the transfer learning and save the trained model (import path of trained model to `app.py`). 
