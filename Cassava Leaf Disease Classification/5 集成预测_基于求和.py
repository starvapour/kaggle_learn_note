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
on_kaggle = True

if on_kaggle:
    test_path = "../input/cassava-leaf-disease-classification/test_images/"
    model_dict = "../input/leaf-disease-model/"
    output_path = "./"
else:
    test_path = "test_images/"
    model_dict = "trained_models/"
    output_path = ""

eff_model_paths = [model_dict + "save_model_efficientnet_0.pth",
                   model_dict + "save_model_efficientnet_1.pth",
                   model_dict + "save_model_efficientnet_2.pth",
                   model_dict + "save_model_efficientnet_3.pth",
                   model_dict + "save_model_efficientnet_4.pth"]

# Test config
# batch size
batchSize = 16

# ------------------------------------dataset------------------------------------
# create dataset
class Leaf_test_Dataset(Dataset):
    def __init__(self, file_list, test_path, transform):
        # get lists
        self.file_list = file_list
        self.test_path = test_path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.test_path + self.file_list[index]
        img = Image.open(image_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_list)

# ------------------------------------main------------------------------------
# main
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # create dataset
    file_list = None
    for path, dirs, files in os.walk(test_path, topdown=False):
        file_list = list(files)

    # preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Leaf_test_Dataset(file_list, test_path, transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize)

    print("Start testing:")

    # net model
    eff_models = []
    for model_path in eff_model_paths:
        eff_net = EfficientNet.from_name('efficientnet-b4')
        eff_net._fc = nn.Linear(eff_net._fc.in_features, 5)
        eff_net.load_state_dict(torch.load(model_path))
        eff_net = eff_net.to(device)
        eff_net.eval()
        eff_models.append(eff_net)

    result = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        batch_num = len(test_loader)
        for index, image in enumerate(test_loader):
            image = image.to(device)

            eff_result = []
            for eff_net in eff_models:
                if len(eff_result) == 0:
                    eff_result = np.array(softmax(eff_net(image).to('cpu')))

                else:
                    eff_result = eff_result + np.array(softmax(eff_net(image).to('cpu')))

            output = torch.tensor(eff_result, dtype=torch.float32)
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.view(pred.shape[0], -1)
            result = result + list(map(lambda x: int(x), pred))

            if (index + 1) % 10 == 0:
                print("Batch: %4d / %4d" % (index + 1, batch_num))

    pred_result = pd.concat([pd.DataFrame(file_list, columns=['image_id']), pd.DataFrame(result, columns=['label'])], axis=1)
    pred_result.to_csv(output_path + "submission.csv", index=False, sep=',')

    print("Done.")


if __name__ == '__main__':
    main()




