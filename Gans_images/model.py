import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


# 生成器,基于上采样
class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.expand = nn.Sequential(
            nn.Linear(128, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
        )
        self.gen = nn.Sequential(
            # 反卷积扩张尺寸，保持kernel size能够被stride整除来减少棋盘效应
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 尾部添加正卷积压缩减少棋盘效应
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),


            # # 反卷积扩张尺寸
            # nn.ConvTranspose2d(128, 128, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(128, 256, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
            # 将输出约束到[-1,1]
            nn.Tanh()
        )

    def forward(self, img_seeds):
        img_seeds = self.expand(img_seeds)
        # 将线性数据重组为二维图片
        img_seeds = img_seeds.view(-1, 128, 8, 8)
        output = self.gen(img_seeds)
        return output


# 返回对应的生成器
def get_G_model(from_old_model, device, model_path):
    model = G_net()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)

    return model


# 判别器
# class D_net(nn.Module):
#     def __init__(self):
#         super(D_net, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.model = EfficientNet.from_pretrained('efficientnet-b0')
#         self.model._fc = nn.Linear(self.model._fc.in_features, 1)
#
#     def forward(self, img):
#         output = self.model(img)
#         output = self.sigmoid(output)
#         return output

class D_net(nn.Module):
    def __init__(self):
        super(D_net,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        features = self.features(img)
        features = features.view(features.shape[0], -1)
        output = self.classifier(features)
        return output

# class D_net(nn.Module):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(D_net, self).__init__()
#         use_bias = False
#
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, input):
#         """Standard forward."""
#         matrix = self.model(input)
#         matrix = self.sigmoid(matrix)
#         value = torch.tensor(list(map(lambda x: x.mean(), matrix)), dtype=torch.float32).requires_grad_().to('cuda:0')
#         value = value.view((-1, 1))
#         return value

# 返回判别器的模型
def get_D_model(from_old_model, device, model_path):
    model = D_net()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)

    return model
