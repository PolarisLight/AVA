import torch
import torch.nn as nn
import torchvision
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2

class ResNetBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        pass

    def forward(self,x):
        pass

class GraphConvLayer(nn.Module):
    def __init__(self,dim_feature=128,bias=True):
        super(GraphConvLayer, self).__init__()

        self.mapping1 = nn.RNN(input_size=dim_feature,hidden_size=1,num_layers=1)
        self.mapping2 = nn.RNN(input_size=dim_feature,hidden_size=1,num_layers=1)

        self.GCN = nn.Linear


    def forward(self, x):
        h0 = torch.zeros(x.shape[0],1)


class AAM(nn.Module):
    def __init__(self):
        super(AAM, self).__init__()
        sam = sam_model_registry["default"]("sam_vit_h_4b8939.pth")
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        mask = []
        for i in range(x.shape[0]):
            mask.append(self.mask_generator.generate(x[i]))
        # mask = torch.stack(mask, dim=0)
        return mask


if __name__ == "__main__":
    from dataset import AVADataset, train_transform
    from torch.utils.data import DataLoader

    image_dir = "dataset/images"
    train_csv = "dataset/labels/train_labels.csv"

    model = AAM()

    dataset = AVADataset(csv_file=train_csv, root_dir=image_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(dataloader):
        imgs, labels = data["image"], data["annotations"]
        mask = model(imgs)
        for item in mask:
            print(item)
        break