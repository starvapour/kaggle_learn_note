from torch.utils.data import Dataset, DataLoader
import time
from torch.optim import AdamW
from model import *
from torchvision.utils import save_image
import random
from torch.autograd import Variable
import os
import cv2
from albumentations import Normalize, Compose, Resize
from albumentations.pytorch import ToTensorV2
from apex import amp

# ------------------------------------config------------------------------------
class config:
    # 设置种子数，配置是否要固定种子数
    seed = 26
    use_seed = True

    # 配置是否要从磁盘加载之前保存的模式参数继续训练
    from_old_model = False

    # 使用apex加速训练
    use_apex = True

    # 运行多少个epoch之后停止
    epochs = 10000
    # 配置batch size
    batchSize = 32

    # 训练图片输入分辨率
    img_size = 265

    # 配置喂入生成器的随机正态分布种子数有多少维（如果改动，需要在model中修改网络对应参数）
    img_seed_dim = 128

    # 有多大概率在训练判别器D时交换正确图片的标签和伪造图片的标签
    D_train_label_exchange = 0.1

    # 保存模型参数文件的路径
    G_model_path = "G_model.pth"
    D_model_path = "D_model.pth"

    # 损失函数
    # 使用均方差损失函数
    criterion = nn.MSELoss()

    # ------------------------------------路径配置------------------------------------
    # 数据集来源
    img_path = "train_images/"
    # 输出图片的文件夹路径
    output_path = "output_images/"


# 固定随机数种子
def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if config.use_seed:
    seed_all(seed=config.seed)

# -----------------------------------transforms------------------------------------
def get_transforms(img_size):
    # 缩放分辨率并转换到0-1之间
    return Compose(
        [Resize(img_size, img_size),
         Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0),
         ToTensorV2(p=1.0)]
    )

# ------------------------------------dataset------------------------------------
# create dataset
class image_dataset(Dataset):
    def __init__(self, file_list, img_path, transform):
        # files list
        self.file_list = file_list
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.img_path + self.file_list[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img

    def __len__(self):
        return len(self.file_list)


# ------------------------------------main------------------------------------
def main():
    # 如果可以使用GPU运算，则使用GPU，否则使用CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # 创建输出文件夹
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    # 创建dataset
    # create dataset
    file_list = None
    for path, dirs, files in os.walk(config.img_path, topdown=False):
        file_list = list(files)

    train_dataset = image_dataset(file_list, config.img_path, transform=get_transforms(config.img_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batchSize, shuffle=True)

    # 从model中获取判别器D和生成器G的网络模型
    G_model = get_G_model(config.from_old_model, device, config.G_model_path)
    D_model = get_D_model(config.from_old_model, device, config.D_model_path)

    # 定义G和D的优化器，此处使用AdamW优化器
    G_optimizer = AdamW(G_model.parameters(), lr=3e-4, weight_decay=1e-6)
    D_optimizer = AdamW(D_model.parameters(), lr=3e-4, weight_decay=1e-6)

    # 损失函数
    criterion = config.criterion

    # 混合精度加速
    if config.use_apex:
        G_model, G_optimizer = amp.initialize(G_model, G_optimizer, opt_level="O1")
        D_model, D_optimizer = amp.initialize(D_model, D_optimizer, opt_level="O1")

    # 记录训练时间
    train_start = time.time()

    # 开始训练的每一个epoch
    for epoch in range(config.epochs):
        print("start epoch "+str(epoch+1)+":")
        # 定义一些变量用于记录进度和损失
        batch_num = len(train_loader)
        D_loss_sum = 0
        G_loss_sum = 0
        count = 0

        # 从dataloader中提取数据
        for index, images in enumerate(train_loader):
            count += 1
            # 将图片放入运算设备的内存
            images = images.to(device)

            # 定义真标签，使用标签平滑的策略，生成0.9到1之间的随机数作为真实标签
            # real_labels = (1 - torch.rand(config.batchSize, 1)/10).to(device)
            # 定义真标签，全1
            # real_labels = Variable(torch.ones(config.batchSize, 1)).to(device)
            # 定义真标签，全0.9
            real_labels = (Variable(torch.ones(config.batchSize, 1))-0.1).to(device)

            # 定义假标签，单向平滑，因此不对生成器标签进行平滑处理，全0
            fake_labels = Variable(torch.zeros(config.batchSize, 1)).to(device)

            # 将随机的初始数据喂入生成器生成假图像
            img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
            fake_images = G_model(img_seeds)

            # 记录真假标签是否被交换过
            exchange_labels = False

            # 有一定概率在训练判别器时交换label
            if random.uniform(0, 1) < config.D_train_label_exchange:
                real_labels, fake_labels = fake_labels, real_labels
                exchange_labels = True

            # 训练判断器D
            D_optimizer.zero_grad()
            # 用真样本输入判别器
            real_output = D_model(images)

            # 对于数据集末尾的数据，长度不够一个batch size时需要去除过长的真实标签
            if len(real_labels) > len(real_output):
                D_loss_real = criterion(real_output, real_labels[:len(real_output)])
            else:
                D_loss_real = criterion(real_output, real_labels)
            # 用假样本输入判别器
            fake_output = D_model(fake_images)
            D_loss_fake = criterion(fake_output, fake_labels)
            # 将真样本与假样本损失相加，得到判别器的损失
            D_loss = D_loss_real + D_loss_fake
            D_loss_sum += D_loss.item()

            # 重置优化器
            D_optimizer.zero_grad()
            # 用损失更新判别器D
            if config.use_apex:
                with amp.scale_loss(D_loss, D_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                D_loss.backward()
            D_optimizer.step()

            # 如果之前交换过标签，此时再换回来
            if exchange_labels:
                real_labels, fake_labels = fake_labels, real_labels

            # 训练生成器G
            # 将随机种子数喂入生成器G生成假数据
            img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
            fake_images = G_model(img_seeds)
            # 将假数据输入判别器
            fake_output = D_model(fake_images)
            # 将假数据的判别结果与真实标签对比得到损失
            G_loss = criterion(fake_output, real_labels)
            G_loss_sum += G_loss.item()

            # 重置优化器
            G_optimizer.zero_grad()
            # 利用损失更新生成器G
            if config.use_apex:
                with amp.scale_loss(G_loss, G_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                G_loss.backward()
            G_optimizer.step()

            # 打印程序工作进度
            if (index + 1) % 200 == 0:
                print("Epoch: %2d, Batch: %4d / %4d" % (epoch + 1, index + 1, batch_num))

        if (epoch+1) % 10 == 0:
            # 在每N个epoch结束时保存模型参数到磁盘文件
            torch.save(G_model.state_dict(), config.G_model_path)
            torch.save(D_model.state_dict(), config.D_model_path)
            # 在每N个epoch结束时输出一组生成器产生的图片到输出文件夹
            img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
            fake_images = G_model(img_seeds).cuda().data
            # 将假图像缩放到[0,1]的区间
            fake_images = 0.5 * (fake_images + 1)
            fake_images = fake_images.clamp(0, 1)
            # 连接所有生成的图片然后用自带的save_image()函数输出到磁盘文件
            fake_images = fake_images.view(-1, 3, config.img_size, config.img_size)
            save_image(fake_images, config.output_path+str(epoch+1)+'.png')

        # 打印该epoch的损失，时间等数据用于参考
        print("D_loss:", round(D_loss_sum / count, 3))
        print("G_loss:", round(G_loss_sum / count, 3))
        current_time = time.time()
        pass_time = int(current_time - train_start)
        time_string = str(pass_time // 3600) + " hours, " + str((pass_time % 3600) // 60) + " minutes, " + str(
            pass_time % 60) + " seconds."
        print("Time pass:", time_string)
        print()

    # 运行结束
    print("Done.")


if __name__ == '__main__':
    main()
