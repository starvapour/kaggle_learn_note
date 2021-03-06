import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as toptim
from torch.utils.data import Dataset, DataLoader
import time
from Loss import LabelSmoothingLoss, FocalLoss
from album_transform import get_train_transforms,get_test_transforms
from model import get_model
from apex import amp
from torch.optim import AdamW
import numpy as np
import sys
sys.path.append("FMix-master")
from fmix import sample_mask, make_low_freq_image, binarise_mask



# ------------------------------------config------------------------------------
class config:
    # all the seed
    seed = 26
    use_seed = True

    # input image size
    img_size = 512

    # use which model
    model_name = "efficientnet"
    # model_name = "resnet50"
    # model_name = "resnext50_32x4d"
    # model_name = "custom"

    # continue train from old model, if not, load pretrained data
    from_old_model = False

    # whether use apex or not
    use_apex = True

    # if true, remove noise in noise.txt in train dataset
    remove_noise = False

    # whether only train output layer
    only_train_output_layer = False
    # when only train output layer, use small train set for fast train
    fast_train_output_layer = False

    # if true, do more pre-processing to change the image
    use_image_enhancement = True

    # if true, use cutmix
    use_cutmix = False

    # if true, use fmix
    use_fmix = True

    # learning rate
    learning_rate = 3e-4
    # max epoch
    epochs = 50
    # batch size
    batchSize = 8

    # if acc is more than this value, start save model
    lowest_save_acc = 0

    # loss function
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(classes=5, smoothing=0.1)
    #criterion = FocalLoss()
    # 多标签
    criterion = nn.MultiLabelSoftMarginLoss()

    # create optimizer
    # optimizer_name = "SGD"
    # optimizer_name = "Adam"
    optimizer_name = "AdamW"

    # Use how many data of the dataset for val, not used now
    # proportion_of_val_dataset = 0.2

    # the index of each 0.2 part, which part used for val, form 0 to 4
    val_index = 1

    # model output
    output_channel = 5

    # read data from where
    # read_data_from = "Memory"
    read_data_from = "Disk"

    # ------------------------------------path set------------------------------------
    train_csv_path = "train.csv"
    noise_path = "noise.txt"

    train_image = "train_images/"
    log_name = "log.txt"
    model_path = "trained_models/save_model_" + model_name + "_" + str(val_index) + ".pth"

# record best val acc with (epoch_num, last_best_acc)
best_val_acc = (-1, config.lowest_save_acc)

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if config.use_seed:
    seed_torch(seed=config.seed)

# ------------------------------------dataset------------------------------------
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if config.read_data_from == "Memory":
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

elif config.read_data_from == "Disk":
    # create dataset
    class Leaf_train_Dataset(Dataset):
        def __init__(self, data_csv, img_path, transform, use_cutmix, use_fmix):
            # get lists
            self.csv = data_csv
            self.img_path = img_path
            self.transform = transform
            self.use_cutmix = use_cutmix
            self.use_fmix = use_fmix

        def __getitem__(self, index):
            image_id = self.csv.loc[index, 'image_id']
            label = self.csv.loc[index, 'label']
            img = cv2.imread(self.img_path + image_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']

            ori_label = label
            label = np.zeros(5)
            label[ori_label] = 1.
            if self.use_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
                with torch.no_grad():
                    cmix_ix = np.random.choice(self.csv.index, size=1)[0]
                    cmix_img = cv2.cvtColor(cv2.imread(self.img_path + self.csv.loc[cmix_ix, 'image_id']), cv2.COLOR_BGR2RGB)
                    cmix_img = self.transform(image=cmix_img)['image']

                    lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
                    bbx1, bby1, bbx2, bby2 = rand_bbox((config.img_size,config.img_size), lam)
                    rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (
                            config.img_size * config.img_size))
                    img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]
                    label[ori_label] = rate
                    label[self.csv.loc[cmix_ix, 'label']] = (1. - rate)

            if self.use_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
                with torch.no_grad():
                    lam = np.clip(np.random.beta(1, 1), 0.6, 0.7)
                    mask = make_low_freq_image(3, (config.img_size,config.img_size))
                    mask = binarise_mask(mask, lam, (config.img_size,config.img_size), True)
                    fmix_ix = np.random.choice(self.csv.index, size=1)[0]
                    fmix_img = cv2.cvtColor(cv2.imread(self.img_path + self.csv.loc[fmix_ix, 'image_id']),
                                            cv2.COLOR_BGR2RGB)
                    fmix_img = self.transform(image=fmix_img)['image']
                    mask_torch = torch.from_numpy(mask)
                    rate = mask.sum() / config.img_size / config.img_size

                    mask_torch = torch.as_tensor(mask_torch, dtype=torch.float32)
                    img = mask_torch * img + (1. - mask_torch) * fmix_img
                    label[ori_label] = rate
                    label[self.csv.loc[fmix_ix, 'label']] = (1. - rate)

            return img, label

        def __len__(self):
            return len(self.csv)

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
        softmax = nn.Softmax(dim=1)
        output = net(imgs)

        # calculate loss
        #output = output.reshape(target.shape)
        loss = criterion(output, labels.long())

        runningLoss += loss.item()
        loss_count += 1

        # calculate gradients.
        if config.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # reduce loss
        optimizer.step()

        # print loss
        # print(index)
        if (index + 1) % 400 == 0:
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
            pred = output.argmax(dim=1, keepdim=True).flatten()
            labels = labels.argmax(dim=1, keepdim=True).flatten()
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
            torch.save(net.state_dict(), config.model_path)
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
    original_csv_data = pd.read_csv(config.train_csv_path)
    print("length of original dataset is", len(original_csv_data))
    log.write("length of original dataset is " + str(len(original_csv_data)) + "\n")

    # split dataset, get train and val
    part_len = int(len(original_csv_data) * 0.2)
    if config.val_index != 4:
        train_csv = pd.concat([original_csv_data.iloc[:config.val_index * part_len], original_csv_data.iloc[(config.val_index+1) * part_len:]],axis=0,join='inner')
    else:
        train_csv = original_csv_data.iloc[:config.val_index * part_len]
    train_csv = train_csv.reset_index(drop=True)
    val_csv = original_csv_data.iloc[config.val_index * part_len:(config.val_index+1) * part_len]
    val_csv = val_csv.reset_index(drop=True)

    # for fast train output layer only
    if config.only_train_output_layer and config.fast_train_output_layer:
        train_csv = train_csv[:1000]
        val_csv = val_csv[:200]

    # remove noise data in noise file
    if config.remove_noise:
        with open(config.noise_path) as noise_file:
            noise = noise_file.readline().split(",")
            # remove noise in training data
            delete_index = []
            for index in range(len(train_csv)):
                if train_csv.loc[index, 'image_id'] in noise:
                    delete_index.append(index)
            train_csv = train_csv.drop(delete_index)
            train_csv = train_csv.reset_index(drop=True)

    print("Start load train dataset:")
    if config.use_image_enhancement:
        train_dataset = Leaf_train_Dataset(train_csv, config.train_image, transform=get_train_transforms(config.img_size), use_cutmix = config.use_cutmix, use_fmix = config.use_fmix)
    else:
        train_dataset = Leaf_train_Dataset(train_csv, config.train_image, transform=get_test_transforms(config.img_size), use_cutmix = config.use_cutmix, use_fmix = config.use_fmix)
    print("length of train dataset is", len(train_dataset))
    log.write("length of train dataset is " + str(len(train_dataset)) + "\n")

    print("Start load val dataset:")
    val_dataset = Leaf_train_Dataset(val_csv, config.train_image, transform=get_test_transforms(config.img_size), use_cutmix = False, use_fmix = False)
    print("length of val dataset is", len(val_dataset))
    log.write("length of val dataset is " + str(len(val_dataset)) + "\n\n")

    print()
    print("Start training:")

    # create dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batchSize, shuffle=True)

    # net model
    net = get_model(config.model_name, config.from_old_model, device, config.model_path, config.output_channel)

    # if only train output layer
    if config.only_train_output_layer:
        for name, value in net.named_parameters():
            if name != "_fc.weight" and name != "_fc.bias" and name != "fc.weight" and name != "fc.bias":
                value.requires_grad = False
        # setup optimizer
        params = filter(lambda x: x.requires_grad, net.parameters())
    else:
        params = net.parameters()

        # create optimizer
    if config.optimizer_name == "SGD":
        optimizer = toptim.SGD(params, lr=config.learning_rate)
    elif config.optimizer_name == "Adam":
        optimizer = toptim.Adam(params, lr=config.learning_rate)
    elif config.optimizer_name == "AdamW":
        optimizer = AdamW(params, lr=config.learning_rate, weight_decay=1e-6)

    # 混合精度加速
    if config.use_apex:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    train_start = time.time()

    for epoch in range(config.epochs):
        '''
        # change lr by epoch
        adjust_learning_rate(optimizer, epoch)
        '''

        # start train
        train(net, train_loader, config.criterion, optimizer, epoch, device, log)

        # start val
        val(net, val_loader, config.criterion, optimizer, epoch, device, log, train_start)

    print("Final saved model is epoch "+str(best_val_acc[0])+", acc: "+str(best_val_acc[1])+".")
    log.write("Final saved model is epoch "+str(best_val_acc[0])+", acc: "+str(best_val_acc[1])+"\n")

    print("Done.")
    log.write("Done.\n")


if __name__ == '__main__':
    with open(config.log_name, 'w') as log:
        main()




