# issues with data collection, needed to make this to make all names and numbering consistent
import os

path = "C:/Users/Evan/OneDrive - University of Waterloo/Desktop/MLProjects/FaceDetectionApp/classifier/dataset/validation/not_me"
files = os.listdir(path)

for index, file in enumerate(files):
    os.rename(
        os.path.join(path, file),
        os.path.join(path, "".join([f"detected_{str(index)}", ".jpg"])),
    )
