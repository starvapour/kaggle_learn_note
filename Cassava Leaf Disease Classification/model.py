from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch

old_data_path = "save_model.pth"
output_channel = 10

# get model
def get_model(model_name, from_old_model, device):

    if model_name == "efficientnet-b5":
        if from_old_model:
            net = EfficientNet.from_name('efficientnet-b5')
            net._fc.out_features = output_channel
            net.load_state_dict(torch.load(old_data_path))
            net = net.to(device)
        else:
            net = EfficientNet.from_pretrained('efficientnet-b5')
            net._fc.out_features = output_channel
            net = net.to(device)

    elif model_name == "resnet50":
        if from_old_model:
            net = models.resnet50(pretrained=False)
            net.fc.out_features = output_channel
            net.load_state_dict(torch.load(old_data_path))
            net = net.to(device)
        else:
            net = models.resnet50(pretrained=True)
            net.fc.out_features = output_channel
            net = net.to(device)

    return net