import torch
import torch.nn as nn
import torchvision
import tqdm
import torch.nn.functional as F


class convblock(nn.Module):
    """

    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(convblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class upblock(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super(upblock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, scale=3):
        super(UNet, self).__init__()
        self.scale = scale
        self.out_channel = out_channel
        base_channel = 64
        self.features = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8,
                         base_channel * 16]  # [64, 128, 256, 512, 1024]
        self.down_block1 = convblock(in_channel, self.features[0])  # 3,512,512 -> 64,512,512
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64,512,512 -> 64,256,256
        self.down_block2 = convblock(self.features[0], self.features[1])  # 64,256,256 -> 128,256,256
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128,256,256 -> 128,128,128
        self.down_block3 = convblock(self.features[1], self.features[2])  # 128,128,128 -> 256,128,128
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256,128,128 -> 256,64,64

        self.bridge = convblock(self.features[2], self.features[3])  # 256,64,64 -> 512,64,64

        self.up_block1 = upblock(self.features[3], self.features[2])  # 512,64,64 -> 256,128,128
        self.conv_block1 = convblock(self.features[2] + self.features[2],
                                     self.features[2])  # 2*256,128,128 -> 256,128,128
        self.up_block2 = upblock(self.features[2], self.features[1])  # 256,128,128 -> 128,256,256
        self.conv_block2 = convblock(self.features[1] + self.features[1],
                                     self.features[1])  # 2*128,256,256 -> 128,256,256
        self.up_block3 = upblock(self.features[1], self.features[0])  # 128,256,256 -> 64,512,512
        self.conv_block3 = convblock(self.features[0] + self.features[0],
                                     self.features[0])  # 2*64,512,512 -> 64,512,512
        self.out = nn.Conv2d(self.features[0], out_channel, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.down_block1(x)  # 3,512,512 -> 64,512,512
        x2 = self.downsample1(x1)  # 64,512,512 -> 64,256,256
        x2 = self.down_block2(x2)  # 64,256,256 -> 128,256,256
        x3 = self.downsample2(x2)  # 128,256,256 -> 128,128,128
        x3 = self.down_block3(x3)  # 128,128,128 -> 256,128,128
        x4 = self.downsample3(x3)  # 256,128,128 -> 256,64,64

        x4 = self.bridge(x4)  # 256,64,64 -> 512,64,64

        x4 = self.up_block1(x4)  # 512,64,64 -> 256,128,128
        x3 = torch.cat([x3, x4], dim=1)  # [256,128,128;256,128,128] -> 512,128,128
        x3 = self.conv_block1(x3)  # 512,128,128 -> 256,128,128
        x3 = self.up_block2(x3)  # 256,128,128 -> 128,256,256
        x2 = torch.cat([x2, x3], dim=1)  # [128,256,256;128,256,256] -> 256,256,256
        x2 = self.conv_block2(x2)  # 256,256,256 -> 128,256,256
        x2 = self.up_block3(x2)  # 128,256,256 -> 64,512,512
        x1 = torch.cat([x1, x2], dim=1)  # [64,512,512;64,512,512] -> 128,512,512
        x1 = self.conv_block3(x1)  # 128,512,512 -> 64,512,512

        if self.out_channel != self.features[0]:
            x1 = self.out(x1)

        return x1


class GraphConvLayer(nn.Module):
    def __init__(self, dim_feature=64, bias=False):
        super(GraphConvLayer, self).__init__()

        self.mapping1 = nn.Linear(dim_feature, dim_feature, bias=bias)
        self.mapping2 = nn.Linear(dim_feature, dim_feature, bias=bias)

        self.GCN_W = nn.Parameter(torch.FloatTensor(dim_feature, dim_feature))
        # self.GCN_B = nn.Parameter(torch.FloatTensor(dim_feature))
        self.relu = nn.ReLU(inplace=True)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.GCN_W)
        # nn.init.xavier_uniform_(self.GCN_B)

    def forward(self, x):
        m1 = self.mapping1(x)
        m2 = self.mapping2(x)

        similarity = torch.matmul(m1, m2.transpose(1, 2))
        similarity = F.softmax(similarity, dim=2)

        x = torch.matmul(similarity, x)
        x = torch.matmul(x, self.GCN_W)
        # x = x + self.GCN_B
        x = self.relu(x)

        return x


class AAM(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM, self).__init__()
        self.feature_extractor = UNet(in_channel=3, out_channel=feat_num)
        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_num * mask_num, feat_num, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feat_num, out_class),
            nn.Softmax(dim=1)
        )

    def forward(self, imgs, masks):
        feats = self.feature_extractor(imgs)
        masked_features = feats.unsqueeze(1) * masks.unsqueeze(2)

        # 计算全局平均池化
        pooled_features = F.adaptive_avg_pool2d(masked_features, (1, 1)).squeeze(-1).squeeze(-1)

        gcn1 = self.GCN_layer1(pooled_features)
        gcn2 = self.GCN_layer2(gcn1)

        pred = self.projector(gcn2)

        return pred


if __name__ == "__main__":
    from dataset import AVADataset, train_transform
    from torch.utils.data import DataLoader
    from utils import EMD_loss

    image_dir = "dataset/images"
    train_csv = "dataset/labels/train_labels.csv"

    model = AAM()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = "cpu"

    model.to(device)

    dataset = AVADataset(csv_file=train_csv, root_dir=image_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = EMD_loss()

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        imgs, labels, masks = data["image"], data["annotations"], data["masks"]
        imgs = imgs.to(device)
        masks = masks.to(device)
        # labels = labels.to(device)

        y = model(imgs, masks)
        loss = criterion(y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i > 10:
            break
