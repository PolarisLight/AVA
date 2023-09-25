import torch
import torch.nn as nn
import torchvision
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
from fastsam import FastSAM


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        pass

    def forward(self, x):
        pass


class GraphConvLayer(nn.Module):
    def __init__(self, dim_feature=128, bias=True):
        super(GraphConvLayer, self).__init__()

        self.mapping1 = nn.RNN(input_size=dim_feature, hidden_size=1, num_layers=1)
        self.mapping2 = nn.RNN(input_size=dim_feature, hidden_size=1, num_layers=1)

        self.GCN = nn.Linear

    def forward(self, x):
        h0 = torch.zeros(x.shape[0], 1)


class AAM(nn.Module):
    def __init__(self, device="mps"):
        super(AAM, self).__init__()
        self.device = device
        self.sam_model = FastSAM('./FastSAM-x.pt')
        basemodel = torchvision.models.resnet18(pretrained=True)
        # print(list(basemodel.children()))
        self.feature_extractor = nn.Sequential(*list(basemodel.children())[:-3])

    def forward(self, x):
        masks = []
        feat = self.feature_extractor(x)
        for i in range(feat.shape[0]):
            # print(x[i].permute(1, 2, 0))
            mask = self.sam_model(x[i].permute(1, 2, 0).numpy(), device=self.device,
                                  retina_masks=True, imgsz=448, conf=0.2, iou=0.9)
            mask = mask[0].masks.data
            masks.append(mask)
        print(feat.shape)
        for item in masks:
            print(item.shape)
        show_img = x[0].permute(1, 2, 0).numpy()
        show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        show_img = cv2.resize(show_img, (448, 448))
        mask = masks[0][0].cpu().numpy()
        show_img = show_img[:,:,0] * mask
        show_img = show_img.astype(np.uint8)
        print(show_img)
        cv2.imshow("test.jpg", show_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    from dataset import AVADataset, train_transform
    from torch.utils.data import DataLoader

    image_dir = "dataset/images"
    train_csv = "dataset/labels/train_labels.csv"

    model = AAM()

    dataset = AVADataset(csv_file=train_csv, root_dir=image_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    for i, data in enumerate(dataloader):
        imgs, labels = data["image"], data["annotations"]

        mask = model(imgs)
        break
