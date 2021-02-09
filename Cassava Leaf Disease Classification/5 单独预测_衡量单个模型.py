import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import sys
sys.path.insert(0, "../input/efficientnet-pytorch")
from efficientnet_pytorch import EfficientNet



# ------------------------------------config------------------------------------
test_path = "../input/cassava-leaf-disease-classification/test_images/"
model_path = "../input/leaf-disease-model/save_model.pth"
output_path = "./"

# Test config
# batch size
batchSize = 16


# ------------------------------------dataset------------------------------------
# create dataset
class Leaf_test_Dataset(Dataset):
    def __init__(self, file_list, test_path, transform):
        # get lists
        self.file_list = file_list
        self.test_path = test_path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.test_path + self.file_list[index]
        img = Image.open(image_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_list)

# ------------------------------------main------------------------------------
# main
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # create dataset
    file_list = None
    for path, dirs, files in os.walk(test_path, topdown=False):
        file_list = list(files)

    # preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Leaf_test_Dataset(file_list, test_path, transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize)

    print("Start testing:")

    # net model
    #net = Leaf_net().to(device)
    net = EfficientNet.from_name('efficientnet-b4')
    net._fc.out_features = 5
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    result = []

    with torch.no_grad():
        batch_num = len(test_loader)
        for index, image in enumerate(test_loader):
            image = image.to(device)
            output = net(image)
            pred = output.argmax(dim=1, keepdim=True)

            pred = pred.view(pred.shape[0], -1)
            result = result + list(map(lambda x:int(x), pred))

            if (index + 1) % 10 == 0:
                print("Batch: %4d / %4d" % (index + 1, batch_num))

    pred_result = pd.concat([pd.DataFrame(file_list, columns=['image_id']), pd.DataFrame(result, columns=['label'])], axis=1)
    pred_result.to_csv(output_path + "submission.csv", index=False, sep=',')

    print("Done.")


if __name__ == '__main__':
    main()




