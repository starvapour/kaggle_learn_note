from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch
import torch.nn as nn
import timm

# get model
def get_model(model_name, from_old_model, device, model_path, output_channel):

    if model_name == "efficientnet":
        if from_old_model:
            net = EfficientNet.from_name('efficientnet-b4')
            #net._fc.out_features = output_channel
            net._fc = nn.Linear(net._fc.in_features, output_channel)
            net.load_state_dict(torch.load(model_path))
            net = net.to(device)
        else:
            net = EfficientNet.from_pretrained('efficientnet-b4')
            # net._fc.out_features = output_channel
            net._fc = nn.Linear(net._fc.in_features, output_channel)
            net = net.to(device)

    elif model_name == "resnet50":
        if from_old_model:
            net = models.resnet50(pretrained=False)
            #net.fc.out_features = output_channel
            net.fc = nn.Linear(net.fc.in_features, output_channel)
            net.load_state_dict(torch.load(model_path))
            net = net.to(device)
        else:
            net = models.resnet50(pretrained=True)
            # net.fc.out_features = output_channel
            net.fc = nn.Linear(net.fc.in_features, output_channel)
            net = net.to(device)

    elif model_name == "resnext50_32x4d":
        class CustomResNext(nn.Module):
            def __init__(self, model_name='resnext50_32x4d', pretrained=False):
                super().__init__()
                self.model = timm.create_model(model_name, pretrained=pretrained)
                n_features = self.model.fc.in_features
                self.model.fc = nn.Linear(n_features, output_channel)

            def forward(self, x):
                x = self.model(x)
                return x

        if from_old_model:
            net = CustomResNext()
            net.load_state_dict(torch.load(model_path))
            net = net.to(device)
        else:
            net = CustomResNext()
            net = net.to(device)

    elif model_name == "custom":
        class CustomResNext(nn.Module):
            def __init__(self):
                super().__init__()
                self.res_model = timm.create_model('resnext50_32x4d', pretrained=False)
                self.eff_model = EfficientNet.from_pretrained('efficientnet-b4')
                self.classifier = nn.Sequential(
                    nn.Linear(1000, 1000),
                    nn.Dropout(0.6),
                    nn.Linear(1000, output_channel),
                )

            def forward(self, x):
                x = self.res_model(x) + self.eff_model(x)
                x = self.classifier(x)

                return x

        if from_old_model:
            net = CustomResNext()
            net.load_state_dict(torch.load(model_path))
            net = net.to(device)
        else:
            net = CustomResNext()
            net = net.to(device)
    return net