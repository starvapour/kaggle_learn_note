import cv2
import torch
import os
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
from efficientnet_pytorch import EfficientNet
from LabelSmoothingLoss import LabelSmoothingLoss
from album_transform import get_train_transforms,get_test_transforms
from model import get_model



# ------------------------------------config------------------------------------
# use which model
#model_name = "efficientnet-b5"
model_name = "resnet50"

# continue train from old model, if not, load pretrain data
from_old_model = True

# learning rate
learning_rate = 3e-4
# max epoch
epochs = 20
# batch size
batchSize = 16

# if acc is more than this value, start save model
lowest_save_acc = 0

# loss function
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)

# create optimizer
#optimizer_name = "SGD"
optimizer_name = "Adam"

# Use how many data of the dataset for val
proportion_of_val_dataset = 0.3

# model output
output_channel = 10

# ------------------------------------other set------------------------------------
train_csv_path = "train.csv"
train_image = "train_images/"
log_name = "log.txt"
model_path = "save_model.pth"

# record best val acc with (epoch_num, last_best_acc)
best_val_acc = (-1, lowest_save_acc)

# ------------------------------------dataset------------------------------------
# create dataset
class Leaf_train_Dataset(Dataset):
    def __init__(self, data_csv, img_path, transform):
        # get lists
        self.transform = transform
        self.data = []

        for index in range(len(data_csv)):
            image_id = data_csv.loc[index, 'image_id']
            label = data_csv.loc[index, 'label']
            img = cv2.imread(img_path + image_id)
            #img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.data.append((img, label))

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transform(image=img)['image']
        return img, label

    def __len__(self):
        return len(self.data)

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
def val(net, val_loader, criterion, optimizer, epoch, device, log, train_start):
    # val after each epoch
    net.eval()

    with torch.no_grad():
        total_len = 0
        correct_len = 0
        global best_val_acc
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

        # if have better acc, save model, only keep the best model in epochs
        if accuracy > best_val_acc[1]:
            # save model
            best_val_acc = (epoch+1, accuracy)
            torch.save(net.state_dict(), model_path)
            print("Model saved in epoch "+str(epoch+1)+", acc: "+str(accuracy)+".")
            log.write("Model saved in epoch "+str(epoch+1)+", acc: "+str(accuracy)+".\n")


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
    original_csv_data = pd.read_csv(train_csv_path)
    print("length of original dataset is", len(original_csv_data))
    log.write("length of original dataset is " + str(len(original_csv_data)) + "\n")

    # split dataset, get train and val
    train_len = int((1 - proportion_of_val_dataset) * len(original_csv_data))
    train_csv = original_csv_data.iloc[:train_len]
    val_csv = original_csv_data.iloc[train_len:]
    val_csv = val_csv.reset_index(drop=True)

    print("Start load train dataset:")
    train_dataset = Leaf_train_Dataset(train_csv, train_image, transform=get_train_transforms())
    print("length of train dataset is", len(train_dataset))
    log.write("length of train dataset is " + str(len(train_dataset)) + "\n")

    print("Start load val dataset:")
    val_dataset = Leaf_train_Dataset(val_csv, train_image, transform=get_test_transforms())
    print("length of val dataset is", len(val_dataset))
    log.write("length of val dataset is " + str(len(val_dataset)) + "\n\n")

    print()
    print("Start training:")

    # create dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batchSize, shuffle=True)

    # net model
    net = get_model(model_name, from_old_model, device, model_path, output_channel)

    # create optimizer
    if optimizer_name == "SGD":
        optimizer = toptim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = toptim.Adam(net.parameters(), lr=learning_rate)

    train_start = time.time()

    for epoch in range(epochs):
        '''
        # change lr by epoch
        adjust_learning_rate(optimizer, epoch)
        '''

        # start train
        train(net, train_loader, criterion, optimizer, epoch, device, log)

        # start val
        val(net, val_loader, criterion, optimizer, epoch, device, log, train_start)

    print("Final saved model is epoch "+str(best_val_acc[0])+", acc: "+str(best_val_acc[1])+".")
    log.write("Final saved model is epoch "+str(best_val_acc[0])+", acc: "+str(best_val_acc[1])+"\n")

    print("Done.")
    log.write("Done.\n")


if __name__ == '__main__':
    with open(log_name, 'w') as log:
        main()




