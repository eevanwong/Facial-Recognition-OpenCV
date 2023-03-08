import cv2 as cv
from facenet_pytorch import MTCNN
import numpy as np

mtcnn = MTCNN()

# Load Vid
v_cap = cv.VideoCapture("./video2.mp4")

# get frame count
v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))

frames = []

print(v_len)

for _ in range(v_len):
    success, frame = v_cap.read()

    if not success:
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frames.append(frame)

save_paths = [f"./dataset/train/me/image_{i+285}.jpg" for i in range(len(frames))]

for frame, path in zip(frames, save_paths):
    try:
        mtcnn(frame, save_path=path)
    except Exception as e:
        print(e)
        pass
