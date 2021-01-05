import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torchvision
import torch.nn as nn
import sys
sys.path.insert(0, "../input/efficientnet-pytorch")
from efficientnet_pytorch import EfficientNet
import numpy as np



# ------------------------------------config------------------------------------
test_path = "test_images/"
efficient_model = "effcientnet_model.pth"
efficient_rate = 0.34
res_model = "resnet_model.pth"
res_rate = 1 - efficient_rate
output_path = ""

# Test config
# batch size
batchSize = 16


# the net model
class combine_net(nn.Module):

    def __init__(self):
        super(combine_net, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2000, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, features):
        # image
        features = features.view(features.shape[0], -1)
        output = self.classifier(features)

        return output

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
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Leaf_test_Dataset(file_list, test_path, transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize)

    print("Start testing:")

    # net model
    # net model
    efficient_net = EfficientNet.from_name('efficientnet-b0')
    efficient_net._fc.out_features = 10
    efficient_net = efficient_net.to(device)
    efficient_net.load_state_dict(torch.load(efficient_model))

    res_net = torchvision.models.resnet50(pretrained=False)
    res_net.fc.out_features = 10
    res_net = res_net.to(device)
    res_net.load_state_dict(torch.load(res_model))

    efficient_net.eval()
    res_net.eval()

    result = []

    with torch.no_grad():
        batch_num = len(test_loader)
        for index, image in enumerate(test_loader):
            image = image.to(device)
            efficient_result = efficient_net(image).to('cpu')
            res_result = res_net(image).to('cpu')
            output = np.array(efficient_result) * efficient_rate + np.array(res_result) * res_rate
            output = torch.tensor(output, dtype=torch.float32)
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.view(pred.shape[0], -1)
            result = result + list(map(lambda x:int(x), pred))

            if (index + 1) % 10 == 0:
                print("Batch: %4d / %4d" % (index + 1, batch_num))

    pred_result = pd.concat([pd.DataFrame(file_list, columns=['image_id']), pd.DataFrame(result, columns=['label'])], axis=1)
    pred_result.to_csv(output_path + "submission.csv", index=False, sep=',')

    print("Done.")


if __name__ == '__main__':
    main()




