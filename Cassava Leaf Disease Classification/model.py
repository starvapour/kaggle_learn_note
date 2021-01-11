from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch

# get model
def get_model(model_name, from_old_model, device, model_path, output_channel):

    if model_name == "efficientnet-b5":
        if from_old_model:
            net = EfficientNet.from_name('efficientnet-b5')
            net._fc.out_features = output_channel
            net.load_state_dict(torch.load(model_path))
            net = net.to(device)
        else:
            net = EfficientNet.from_pretrained('efficientnet-b5')
            net._fc.out_features = output_channel
            net = net.to(device)

    elif model_name == "resnet50":
        if from_old_model:
            net = models.resnet50(pretrained=False)
            net.fc.out_features = output_channel
            net.load_state_dict(torch.load(model_path))
            net = net.to(device)
        else:
            net = models.resnet50(pretrained=True)
            net.fc.out_features = output_channel
            net = net.to(device)

    return net