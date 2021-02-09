import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
# import torchvision
import torch.nn as nn
import sys
sys.path.insert(0, "../input/efficientnet-pytorch")
from efficientnet_pytorch import EfficientNet
import numpy as np


# ------------------------------------config------------------------------------
train_csv_path = "train.csv"
img_path = "train_images/"
model_dict = "trained_models/"
output_path = ""

eff_model_paths = [model_dict + "save_model_efficientnet_0.pth",
                   model_dict + "save_model_efficientnet_1.pth",
                   model_dict + "save_model_efficientnet_2.pth",
                   model_dict + "save_model_efficientnet_3.pth",
                   model_dict + "save_model_efficientnet_4.pth"]

# Test config
# batch size
batchSize = 8

# ------------------------------------dataset------------------------------------
# create dataset
class Leaf_train_Dataset(Dataset):
    def __init__(self, data_csv, img_path, transform):
        # get lists
        self.csv = data_csv
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.csv.loc[index, 'image_id']
        label = self.csv.loc[index, 'label']
        image_path = self.img_path + image_id
        img = Image.open(image_path)
        img = self.transform(img)
        return img, image_id, label

    def __len__(self):
        return len(self.csv)

# ------------------------------------main------------------------------------
# main
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # create dataset

    # preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_csv = pd.read_csv(train_csv_path)
    train_dataset = Leaf_train_Dataset(train_csv, img_path, transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize)

    print("Start testing:")

    # net model
    eff_models = []
    for model_path in eff_model_paths:
        eff_net = EfficientNet.from_name('efficientnet-b4')
        eff_net._fc.out_features = 10
        eff_net.load_state_dict(torch.load(model_path))
        eff_net = eff_net.to(device)
        eff_net.eval()
        eff_models.append(eff_net)

    # create a dataframe from numpy, columns are ["image_id", "label", "pred_0", "pred_1", 'pred_2', "pred_3", "pred_4"]
    preds = []

    with torch.no_grad():
        batch_num = len(train_loader)
        for index, (imgs, image_ids, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            eff_result = [image_ids, labels]
            for eff_net in eff_models:
                output = eff_net(imgs)
                output = output.to('cpu')
                pred = output.argmax(dim=1, keepdim=True).reshape(labels.shape)
                eff_result.append(pred)

            if len(preds) == 0:
                preds = np.dstack(eff_result)[0]
            else:
                preds = np.vstack([preds, np.dstack(eff_result)[0]])

            if (index + 1) % 10 == 0:
                print("Batch: %4d / %4d" % (index + 1, batch_num))

    preds_csv = pd.DataFrame(preds, columns=["image_id", "label", "pred_0", "pred_1", 'pred_2', "pred_3", "pred_4"])
    preds_csv.to_csv(output_path + "models_pred.csv", index=False, sep=',')
    #print(preds)

    print("Done.")


if __name__ == '__main__':
    main()




