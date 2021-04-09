import cv2
import os

# 数据集来源
img_path = "train_images/"

for path, dirs, files in os.walk(img_path, topdown=False):
    file_list = list(files)
for file in file_list:
    image_path = img_path + file
    img = cv2.imread(image_path, 1)
    bias = (img.shape[1] - img.shape[0]) // 2
    img = img[:, bias:bias+img.shape[0], :]
    cv2.imwrite(image_path, img)
