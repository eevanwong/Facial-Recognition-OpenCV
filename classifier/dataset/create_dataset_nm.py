from facenet_pytorch import MTCNN
from PIL import Image
import os, os.path

# grab pictures from dataset on kaggle (or wherever)

path = "./unclean_data/"

mtcnn = MTCNN()

num_files = len([name for name in os.listdir(path)])
print(num_files)
save_paths = [f"./dataset/train/not_me/detected_{i}.jpg" for i in range(num_files)]

for file, new_path in zip(sorted(os.listdir(path)), save_paths):
    try:
        if file[-3:] == "jpg":  # we only consider jpg files for now
            im = Image.open(path + file)
            mtcnn(im, save_path=new_path)
    except Exception as e:
        print(e)
        pass
