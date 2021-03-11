import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from torchtext import data
from data_processing import *
from model_CNN import *
from torch.optim import AdamW
from apex import amp
from stopwords import get_stopwords

# ------------------------------------config------------------------------------
class config:
    # all the seed
    seed = 26
    use_seed = True

    # input image size
    img_size = 512

    # use which model
    model_name = "CNN-LSTM"

    # continue train from old model, if not, load pretrained data
    from_old_model = False

    # whether use apex or not
    use_apex = True

    # whether only train output layer
    only_train_output_layer = False

    # learning rate
    learning_rate = 1e-4
    # max epoch
    epochs = 100
    # batch size
    batchSize = 32

    # if acc is more than this value, start save model
    lowest_save_acc = 0

    # Use how many data of the dataset for val
    proportion_of_val_dataset = 0.2

    # 单词向量化
    dim = 200
    wordVectors = GloVe(name='6B', dim=dim)  # 自定义维度，50，100，200，300

    # loss function
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(classes=5, smoothing=0.1)
    #criterion = FocalLoss()
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.MSELoss()

    # create optimizer
    # optimizer_name = "SGD"
    # optimizer_name = "Adam"
    optimizer_name = "AdamW"

    # model output
    output_channel = 1

    # read data from where
    read_data_from = "Memory"
    # read_data_from = "Disk"

    # ------------------------------------path set------------------------------------
    train_csv_path = "train.csv"

    train_image = "train_images/"
    log_name = "log.txt"
    model_path = "save_model_" + model_name + ".pth"

# record best val acc with (epoch_num, last_best_acc)
best_val_acc = (-1, config.lowest_save_acc)

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if config.use_seed:
    seed_torch(seed=config.seed)


# ------------------------------------train------------------------------------
def train(net, train_loader, criterion, optimizer, epoch, device, log, textField):
    # start train
    runningLoss = 0
    loss_count = 0
    net.train()

    batch_num = len(train_loader)
    for index, batch in enumerate(train_loader):
        # Get a batch and potentially send it to GPU memory.
        inputs = textField.vocab.vectors[batch.text[0]].to(device)
        labels = batch.target.type(torch.FloatTensor).to(device)

        # zero grad
        optimizer.zero_grad()

        # forward
        output = net(inputs)

        # calculate loss
        loss = criterion(output, labels)

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
def val(net, val_loader, criterion, optimizer, epoch, device, log, train_start, textField):
    # val after each epoch
    net.eval()

    with torch.no_grad():
        total_len = 0
        correct_len = 0
        global best_val_acc
        for index, batch in enumerate(val_loader):
            # Get a batch and potentially send it to GPU memory.
            inputs = textField.vocab.vectors[batch.text[0]].to(device)
            labels = batch.target.type(torch.FloatTensor).to(device)
            output = net(inputs)
            pred = (output >= 0.5).flatten()
            labels = labels.flatten()
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
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # Load the training dataset, and create a dataloader to generate a batch.自动处理小写，计算长度
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           preprocessing=preprocessing,  # 单词形式下的预处理，过去式之类的去除
                           postprocessing=postprocessing,
                           stop_words=get_stopwords())  # 剔除stopwords中的所有单词
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

    dataset = data.TabularDataset('train.csv', 'csv',
                                  {'text': ('text', textField),
                                   'target': ('target', labelField)})

    textField.build_vocab(dataset, vectors=config.wordVectors)  # 把数据转换为向量，用上面定义的textfield

    # 分割数据集，训练集与验证集
    train_dataset, validate_dataset = dataset.split(split_ratio=config.proportion_of_val_dataset,
                                        stratified=True, strata_field='target')

    train_loader, val_loader = data.BucketIterator.splits(
            (train_dataset, validate_dataset), shuffle=True, batch_size=config.batchSize,sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = get_model(config.dim, config.from_old_model,config.model_path).to(device)

    criterion = config.criterion

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
        train(net, train_loader, config.criterion, optimizer, epoch, device, log, textField)

        # start val
        val(net, val_loader, config.criterion, optimizer, epoch, device, log, train_start, textField)

    print("Final saved model is epoch "+str(best_val_acc[0])+", acc: "+str(best_val_acc[1])+".")
    log.write("Final saved model is epoch "+str(best_val_acc[0])+", acc: "+str(best_val_acc[1])+"\n")

    print("Done.")
    log.write("Done.\n")

if __name__ == '__main__':
    with open(config.log_name, 'w') as log:
        main()