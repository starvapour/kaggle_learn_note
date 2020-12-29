import cv2
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as toptim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
#import pydicom
import time



# ------------------------------------config------------------------------------
train_csv = "train.csv"
train_image = "train_images/"

# Train config
# learning rate
learning_rate = 0.001
# max epoch
epochs = 100
# batch size
batchSize = 16
# model save each step
model_save_step = 10

# Use how many data of the dataset for val
proportion_of_val_dataset = 0.2

# output path
log_name = "log.txt"

# used for debug, set -1 to use all data
sampler_len = -1

# ------------------------------------dataset------------------------------------
# create dataset，use patient_id, sex,age_approx,anatom_site_general_challenge ,image_name,targets
class Leaf_train_Dataset(Dataset):
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

# ------------------------------------model------------------------------------
# the net model
class Leaf_net(nn.Module):

    def __init__(self):
        super(Leaf_net, self).__init__()

        # use vgg16 model to get features
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 5),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, imgs):
        # image
        features = self.features(imgs)
        features = features.view(features.shape[0], -1)
        output = self.classifier(features)

        return output

# ------------------------------------train------------------------------------
def train(net, train_loader, criterion, optimizer, epoch, device, log):
    # start train
    runningLoss = 0
    loss_count = 0

    batch_num = len(train_loader)
    for index, (imgs, labels) in enumerate(train_loader):
        # send data to device
        imgs, labels = imgs.to(device), labels.to(device)

        # zero grad
        optimizer.zero_grad()

        # forward
        output = net(imgs)

        # calculate loss
        #output = output.reshape(target.shape)
        loss = criterion(output, labels)

        runningLoss += loss.item()
        loss_count += 1

        # calculate gradients.
        loss.backward()

        # reduce loss
        optimizer.step()

        # print loss
        # print(index)
        if (index + 1) % 100 == 0:
            print("Epoch: %2d, Batch: %4d / %4d, Loss: %.3f" % (epoch + 1, index + 1, batch_num, loss.item()))

    avg_loss = runningLoss / loss_count
    print("For Epoch: %2d, Average Loss: %.3f" % (epoch + 1, avg_loss))
    log.write("For Epoch: %2d, Average Loss: %.3f" % (epoch + 1, avg_loss) + "\n")

# ------------------------------------val------------------------------------
def val(net, val_loader, criterion, optimizer, epoch, device, log, val_len, train_start):
    # val after each epoch
    net.eval()

    with torch.no_grad():
        total_len = 0
        correct_len = 0
        for index, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            pred = output.argmax(dim=1, keepdim=True).reshape(labels.shape)
            assessment = torch.eq(pred, labels)
            total_len += len(pred)
            correct_len += int(assessment.sum())
        accuracy = correct_len / total_len
        print("Start val:")
        print("accuracy:", accuracy)
        log.write("accuracy: " + str(accuracy) + "\n")

    # print time pass after each epoch
    current_time = time.time()
    pass_time = int(current_time - train_start)
    time_string = str(pass_time // 3600) + " hours, " + str((pass_time % 3600) // 60) + " minutes, " + str(
        pass_time % 60) + " seconds."
    print("Time pass:", time_string)
    print()
    log.write("Time pass: " + time_string + "\n\n")

# ------------------------------------main------------------------------------
# main
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # create dataset
    original_csv_data = pd.read_csv(train_csv)
    print("length of original dataset is", len(original_csv_data))
    log.write("length of original dataset is " + str(len(original_csv_data)) + "\n")

    # preprocessing steps
    data_transform = transforms.Compose([
        # for vgg16
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    original_dataset = Leaf_train_Dataset(original_csv_data, train_image, transform = data_transform)

    # if need, use smaller dataset for fast debug
    if sampler_len != -1:
        original_dataset, drop_dataset = torch.utils.data.random_split(original_dataset, [sampler_len, len(
            original_dataset) - sampler_len])

    print("length of selected dataset is", len(original_dataset))
    print()
    log.write("length of selected dataset is " + str(len(original_dataset)) + "\n\n")

    # split dataset, get train and val
    train_len = int((1 - proportion_of_val_dataset) * len(original_dataset))
    val_len = len(original_dataset) - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(original_dataset, [train_len, val_len])

    print("Start random split:")
    print("length of train dataset is", len(train_dataset))
    log.write("length of train dataset is " + str(len(train_dataset)) + "\n")
    print("length of val dataset is", len(val_dataset))
    log.write("length of val dataset is " + str(len(val_dataset)) + "\n\n")
    print()
    print("Start training:")

    # create dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batchSize, shuffle=True)

    # net model
    net = Leaf_net().to(device)

    # loss function
    # criterion = tnn.BCEWithLogitsLoss()
    criterion = nn.NLLLoss()

    # create optimizer
    optimizer = toptim.SGD(net.parameters(), lr=learning_rate)
    #optimizer = toptim.Adam(net.parameters(), lr=learning_rate)

    train_start = time.time()

    for epoch in range(epochs):
        '''
        # change lr by epoch
        adjust_learning_rate(optimizer, epoch)
        '''

        # start train
        train(net, train_loader, criterion, optimizer, epoch, device, log)

        # start val
        val(net, val_loader, criterion, optimizer, epoch, device, log, val_len, train_start)
        '''
        # save model after each step
        if (epoch + 1) % model_save_step == 0:
            torch.save(net.state_dict(), save_model_name + "_" + str(epoch + 1) + '.pth')
            print("Model saved.")
            log.write("Model saved.\n")
        '''

    print("Done.")
    log.write("Done.\n")


if __name__ == '__main__':
    with open(log_name, 'w') as log:
        main()




