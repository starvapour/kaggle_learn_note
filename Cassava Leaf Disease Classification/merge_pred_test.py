import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torchvision
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np



# ------------------------------------config------------------------------------
efficient_model = "effcientnet_model.pth"
res_model = "resnet_model.pth"
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
    def __init__(self, data_csv, img_path, transform):
        # get lists
        self.csv = data_csv
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.csv.loc[index, 'image_id']
        label = self.csv.loc[index, 'label']
        img = Image.open(self.img_path + image_id)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.csv)

# ------------------------------------main------------------------------------
# main
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # preprocessing steps
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Leaf_test_Dataset(pd.read_csv("train.csv"), "train_preprocessed_images/", transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize)

    print("Start testing:")

    # net model
    # net model
    efficient_net = EfficientNet.from_name('efficientnet-b5')
    efficient_net._fc.out_features = 10
    efficient_net = efficient_net.to(device)
    efficient_net.load_state_dict(torch.load(efficient_model))

    res_net = torchvision.models.resnet50(pretrained=False)
    res_net.fc.out_features = 10
    res_net = res_net.to(device)
    res_net.load_state_dict(torch.load(res_model))

    efficient_net.eval()
    res_net.eval()

    for efficient_rate in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        res_rate = 1 - efficient_rate
        with torch.no_grad():
            total_len = 0
            correct_len = 0
            batch_num = len(test_loader)
            for index, (imgs, labels) in enumerate(test_loader):
                imgs = imgs.to(device)
                efficient_result = efficient_net(imgs).to('cpu')
                res_result = res_net(imgs).to('cpu')
                output = np.array(efficient_result) * efficient_rate + np.array(res_result) * res_rate
                output = torch.tensor(output, dtype=torch.float32)
                pred = output.argmax(dim=1, keepdim=True).reshape(labels.shape)
                assessment = torch.eq(pred, labels)
                total_len += len(pred)
                correct_len += int(assessment.sum())
                if (index + 1) % 100 == 0:
                    print("Batch: %4d / %4d" % (index + 1, batch_num))
            accuracy = correct_len / total_len
            print(efficient_rate, res_rate, "accuracy: "+str(accuracy) + "\n")
            print()
            log.write("efficient_rate is: "+str(efficient_rate)+"\n")
            log.write("res_rate is: " + str(res_rate) + "\n")
            log.write("accuracy: "+str(accuracy)+"\n\n")

    #pred_result = pd.concat([pd.DataFrame(file_list, columns=['image_id']), pd.DataFrame(result, columns=['label'])], axis=1)
    #pred_result.to_csv(output_path + "submission.csv", index=False, sep=',')

    print("Done.")


if __name__ == '__main__':
    log_name = "log.txt"
    with open(log_name, 'w') as log:
        main()




