import torch
import torch.nn as nn


# 生成器，基于线性层
class G_net_linear(nn.Module):
    def __init__(self):
        super(G_net_linear, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            # 将输出约束到[-1,1]
            nn.Tanh()
        )

    def forward(self, img_seeds):
        output = self.gen(img_seeds)
        # 将线性数据重组为二维图片
        output = output.view(-1, 1, 28, 28)
        return output


# 生成器,基于上采样
class G_net_conv(nn.Module):
    def __init__(self):
        super(G_net_conv, self).__init__()
        # 扩张数据量
        self.expand = nn.Sequential(
            nn.Linear(256, 484),
            nn.BatchNorm1d(484),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(484, 484),
            nn.BatchNorm1d(484),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
        )
        self.gen = nn.Sequential(
            # 反卷积扩张尺寸
            nn.ConvTranspose2d(1, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            # 1x1卷积压缩维度
            nn.Conv2d(4, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            # 将输出约束到[-1,1]
            nn.Tanh()
        )

    def forward(self, img_seeds):
        img_seeds = self.expand(img_seeds)
        # 将线性数据重组为二维图片
        img_seeds = img_seeds.view(-1, 1, 22, 22)
        output = self.gen(img_seeds)
        return output


# 根据生成器的配置返回对应的模型
def get_G_model(from_old_model, device, model_path, G_type):
    if G_type == "Linear":
        model = G_net_linear()
    elif G_type == "ConvTranspose":
        model = G_net_conv()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)

    return model


# 判别器
class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(36864, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # 提取特征
        features = self.features(img)
        # 展平二维矩阵
        features = features.view(features.shape[0],-1)
        # 使用线性层分类
        output = self.classifier(features)
        return output


# 返回判别器的模型
def get_D_model(from_old_model, device, model_path):
    model = D_net()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)

    return model
