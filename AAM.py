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


class resconvblock(nn.Module):
    """

    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        super(resconvblock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True,
                                      groups=groups)

    def forward(self, x):
        out = self.conv(x)
        if self.in_channel != self.out_channel:
            out = out + self.shortcut(x)
        else:
            out = out + x

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
    def __init__(self, in_channel=3, out_channel=32, scale=3, bottleneck=False):
        super(UNet, self).__init__()
        self.scale = scale
        self.out_channel = out_channel
        self.bottleneck = bottleneck
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
        bottleneck = x4

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

        if self.bottleneck:
            return x1, bottleneck
        else:
            return x1


class MyCustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCustomModel, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(2048, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
        )
        # 如果你不需要训练 ResNet50，可以设置为评估模式
        # self.resnet50.eval()
        # 可以移除原始的全连接层，如果你只需要中间特征
        # self.resnet50.fc = nn.Identity()

    def forward(self, x):
        # 提取 layer4 的特征
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        intermediate_features = self.resnet50.layer3(x)
        x = self.resnet50.layer4(intermediate_features)

        # 继续前向传播获取分类结果
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        final_output = self.resnet50.fc(x)

        return final_output, intermediate_features


class FCN(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, scale=4, bias=True, groups=1):
        super(FCN, self).__init__()
        self.scale = scale
        self.out_channel = out_channel
        base_channel = 32
        self.features = [base_channel * (2 ** 0), base_channel * (2 ** 1), base_channel * (2 ** 2),
                         base_channel * (2 ** 3),
                         base_channel * (2 ** 4)]  # [64, 128, 256, 512, 1024]
        self.down_block1 = resconvblock(in_channel, self.features[0], bias=bias,
                                        groups=groups)  # 3,224,224 -> 64,224,224
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64,224,224 -> 64,112,112
        self.down_block2 = resconvblock(self.features[0], self.features[1], bias=bias,
                                        groups=groups)  # 64,112,112 -> 128,112,112
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128,112,112 -> 128,56,56
        self.down_block3 = resconvblock(self.features[1], self.features[2], bias=bias,
                                        groups=groups)  # 128,56,56 -> 256,56,56
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256,56,56 -> 256,28,28
        self.down_block4 = resconvblock(self.features[2], self.features[3], bias=bias,
                                        groups=groups)  # 256,28,28 -> 512,28,28
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512,28,28 -> 512,14,14

        self.bridge = resconvblock(self.features[3], out_channel, bias=bias,
                                   groups=groups)  # 512,14,14 -> out_channel,14,14

    def forward(self, x):
        x = self.down_block1(x)  # 3,512,512 -> 64,512,512
        x = self.downsample1(x)  # 64,512,512 -> 64,256,256
        x = self.down_block2(x)  # 64,256,256 -> 128,256,256
        x = self.downsample2(x)  # 128,256,256 -> 128,128,128
        x = self.down_block3(x)  # 128,128,128 -> 256,128,128
        x = self.downsample3(x)  # 256,128,128 -> 256,64,64
        x = self.down_block4(x)  # 256,64,64 -> 512,64,64
        x = self.downsample4(x)  # 512,64,64 -> 512,32,32

        x = self.bridge(x)  # 512,32,32 -> out_channel,32,32
        return x


class FCN2(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, scale=4, bias=True, groups=1):
        super(FCN2, self).__init__()
        self.scale = scale
        self.out_channel = out_channel
        base_channel = in_channel * 2
        self.features = [base_channel * (2 ** 0),  # 1
                         base_channel * (2 ** 1),  # 2
                         base_channel * (2 ** 2),  # 4
                         base_channel * (2 ** 3),  # 8
                         base_channel * (2 ** 4)]  # 16
        self.down_block1 = resconvblock(in_channel, self.features[0], bias=bias,
                                        groups=groups)  # 3,224,224 -> 64,224,224
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64,224,224 -> 64,112,112
        self.down_block2 = resconvblock(self.features[0], self.features[1], bias=bias,
                                        groups=groups)  # 64,112,112 -> 128,112,112
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128,112,112 -> 128,56,56
        self.down_block3 = resconvblock(self.features[1], self.features[2], bias=bias,
                                        groups=groups)  # 128,56,56 -> 256,56,56
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256,56,56 -> 256,28,28
        self.down_block4 = resconvblock(self.features[2], self.features[3], bias=bias,
                                        groups=groups)  # 256,28,28 -> 512,28,28
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512,28,28 -> 512,14,14

        self.bridge = resconvblock(self.features[3], out_channel * in_channel, bias=bias,
                                   groups=groups)  # 512,14,14 -> out_channel,14,14

    def forward(self, x):
        x = self.down_block1(x)  # 3,512,512 -> 64,512,512
        x = self.downsample1(x)  # 64,512,512 -> 64,256,256
        x = self.down_block2(x)  # 64,256,256 -> 128,256,256
        x = self.downsample2(x)  # 128,256,256 -> 128,128,128
        x = self.down_block3(x)  # 128,128,128 -> 256,128,128
        x = self.downsample3(x)  # 256,128,128 -> 256,64,64
        x = self.down_block4(x)  # 256,64,64 -> 512,64,64
        x = self.downsample4(x)  # 512,64,64 -> 512,32,32

        x = self.bridge(x)  # 512,32,32 -> out_channel,32,32
        return x


class FCN3(nn.Module):
    """
    Changeable scale FCN
    """

    def __init__(self, in_channel=3, out_channel=32, scale=4, bias=True, groups=1):
        super(FCN3, self).__init__()
        self.out_channel = out_channel
        base_channel = 32
        self.features = [base_channel * (2 ** i) for i in range(scale)]  # e.g., [32, 64, 128, 256, ...]

        # Create input block
        self.input_block = resconvblock(in_channel, self.features[0], bias=bias, groups=groups)

        # Create downsample blocks and max pooling layers
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = [self.features[0]] + self.features[1:]
        for i in range(1, scale):
            self.down_blocks.append(resconvblock(in_channels[i - 1], self.features[i], bias=bias, groups=groups))
            self.downsamples.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Create the bridge block
        self.bridge = resconvblock(self.features[-1], out_channel, bias=bias, groups=groups)

    def forward(self, x):
        x = self.input_block(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
            x = self.downsamples[i](x)

        x = self.bridge(x)
        return x


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

    def forward(self, x, A_spa):
        m1 = self.mapping1(x)
        m2 = self.mapping2(x)

        similarity = torch.matmul(m1, m2.transpose(1, 2))
        A_sim = F.softmax(similarity, dim=2)

        A = A_sim + A_spa

        x = torch.matmul(A, x)
        x = torch.matmul(x, self.GCN_W)
        # x = x + self.GCN_B
        x = self.relu(x)

        return x


class AAM(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.feat_num = feat_num
        self.mask_num = mask_num
        self.feature_extractor = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        self.conv1x1 = nn.Conv2d(2048, feat_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)
        if out_class == 1:
            self.gcn_projector = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(feat_num * mask_num, out_class, bias=False),
                nn.Sigmoid()
            )
            self.cnn_projector = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(2048 * 14 * 14, out_class, bias=False),
                nn.Sigmoid()
            )

        else:
            self.gcn_projector = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(feat_num * mask_num, out_class, bias=False),
                nn.Softmax()
            )
            self.cnn_projector = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(2048 * 14 * 14, out_class, bias=False),
                nn.Softmax()
            )

    def forward(self, imgs, masks, mask_loc):
        feats = self.feature_extractor.backbone(imgs)['out']
        if self.feat_num != 2048:
            feats = self.conv1x1(feats)
        feats = F.avg_pool2d(feats, kernel_size=2)
        cnn_pred = self.cnn_projector(feats)

        masks = F.interpolate(masks, size=feats.size()[2:], mode="bilinear", align_corners=False)
        feats = feats.unsqueeze(1) * masks.unsqueeze(2)

        # 计算全局平均池化
        pooled_features = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)

        batch_size, m, _ = mask_loc.size()

        # 计算点之间的欧氏距离
        # 首先扩展张量以进行广播计算
        expanded_points1 = mask_loc.unsqueeze(2).expand(batch_size, m, m, 2)
        expanded_points2 = mask_loc.unsqueeze(1).expand(batch_size, m, m, 2)

        # 计算点之间的欧氏距离
        A_spa = torch.norm(expanded_points1 - expanded_points2, dim=3)
        A_spa = F.softmax(A_spa, dim=2)

        gcn1 = self.GCN_layer1(pooled_features, A_spa)
        gcn2 = self.GCN_layer2(gcn1, A_spa)

        gcn_pred = self.gcn_projector(gcn2)

        pred = (gcn_pred + cnn_pred) / 2

        return pred


class AAM1(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM1, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.feat_num = feat_num
        self.mask_num = mask_num
        self.feature_extractor = FCN(in_channel=3, out_channel=feat_num, bias=False,
                                     groups=1)
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Sequential(
            nn.Linear(2048, out_class),
            nn.Sigmoid() if out_class == 1 else nn.Softmax()
        )

        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)

        self.gcn_projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_num * mask_num, out_class, bias=False),
            nn.Sigmoid() if out_class == 1 else nn.Softmax()
        )

    def forward(self, imgs, masks, mask_loc):
        masked_img = imgs.unsqueeze(1) * masks.unsqueeze(2)  # 2,3,224,224 -> 2,30,3,224,224
        bs = masked_img.shape[0]
        masked_img = torch.reshape(masked_img, (-1, 3,
                                                masked_img.shape[3], masked_img.shape[4]))

        feats = self.feature_extractor(masked_img)
        feats = feats.reshape(bs, self.mask_num, self.feat_num, feats.shape[2],
                              feats.shape[3])  # 2,30,64,28,28

        # 计算全局平均池化
        # feats = feats.view(feats.shape[0], self.mask_num, self.feat_num, feats.shape[2],
        #                    feats.shape[3])

        pooled_features = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
        # 将分割为mask_num组

        # ============计算掩码空间关系============

        batch_size, m, _ = mask_loc.size()

        # 计算点之间的欧氏距离
        # 首先扩展张量以进行广播计算
        expanded_points1 = mask_loc.unsqueeze(2).expand(batch_size, m, m, 2)
        expanded_points2 = mask_loc.unsqueeze(1).expand(batch_size, m, m, 2)

        # 计算点之间的欧氏距离
        A_spa = torch.norm(expanded_points1 - expanded_points2, dim=3)
        A_spa = F.softmax(A_spa, dim=2)
        # =====================================
        gcn1 = self.GCN_layer1(pooled_features, A_spa)
        gcn2 = self.GCN_layer2(gcn1, A_spa)

        gcn_pred = self.gcn_projector(gcn2)
        cnn_pred = self.cnn(imgs)

        pred = (gcn_pred + cnn_pred) / 2

        return pred


class AAM2(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM2, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.feat_num = feat_num
        self.mask_num = mask_num
        self.feature_extractor = FCN2(in_channel=mask_num, out_channel=feat_num, bias=True,
                                      groups=mask_num)
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(2048, out_class),
            nn.Sigmoid() if out_class == 1 else nn.Softmax()
        )

        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)

        self.gcn_projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.75),
            nn.Linear(feat_num * mask_num, out_class, bias=False),
            nn.Sigmoid() if out_class == 1 else nn.Softmax()
        )
        # initial feature extractor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, masks, mask_loc):
        masked_img = imgs.unsqueeze(1) * masks.unsqueeze(2)  # 2,3,224,224 -> 2,30,3,224,224
        bs = masked_img.shape[0]
        masked_img = torch.mean(masked_img, dim=2, keepdim=False)

        feats = self.feature_extractor(masked_img)
        feats = feats.reshape(bs, self.mask_num, -1, feats.shape[2],
                              feats.shape[3])  # 2,30,64,28,28

        pooled_features = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
        # 将分割为mask_num组

        # ============计算掩码空间关系============

        batch_size, m, _ = mask_loc.size()

        # 计算点之间的欧氏距离
        # 首先扩展张量以进行广播计算
        expanded_points1 = mask_loc.unsqueeze(2).expand(batch_size, m, m, 2)
        expanded_points2 = mask_loc.unsqueeze(1).expand(batch_size, m, m, 2)

        # 计算点之间的欧氏距离
        A_spa = torch.norm(expanded_points1 - expanded_points2, dim=3)
        A_spa = F.softmax(A_spa, dim=2)
        # =====================================
        gcn1 = self.GCN_layer1(pooled_features, A_spa)
        gcn2 = self.GCN_layer2(gcn1, A_spa)

        gcn_pred = self.gcn_projector(gcn2)
        cnn_pred = self.cnn(imgs)

        pred = (gcn_pred + cnn_pred) / 2

        return pred


class AAM3(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM3, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.feat_num = feat_num
        self.mask_num = mask_num
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.feature_extractor = FCN3(in_channel=3, out_channel=feat_num, bias=True, scale=4)
        # self.cnn = torchvision.models.resnet50(pretrained=True)
        # self.cnn.fc = nn.Sequential(
        #     nn.Dropout(0.75),
        #     nn.Linear(2048, out_class),
        #     nn.Sigmoid() if out_class == 1 else nn.Softmax()
        # )

        self.feature_extractor = MyCustomModel(num_classes=1)

        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)

        self.gcn_projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.75),
            nn.Linear(feat_num * mask_num, out_class, bias=False),
            nn.Sigmoid() if out_class == 1 else nn.Softmax()
        )
        # initial feature extractor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, masks, mask_loc):
        cnn_pred, feats = self.feature_extractor(imgs)

        for _ in range(4):
            masks = F.max_pool2d(masks, kernel_size=2, stride=2)

        masked_feats = feats.unsqueeze(1) * masks.unsqueeze(2)

        pooled_features = F.adaptive_avg_pool2d(masked_feats, (1, 1)).squeeze(-1).squeeze(-1)
        # 将分割为mask_num组

        # ============计算掩码空间关系============

        batch_size, m, _ = mask_loc.size()

        # 计算点之间的欧氏距离
        # 首先扩展张量以进行广播计算
        expanded_points1 = mask_loc.unsqueeze(2).expand(batch_size, m, m, 2)
        expanded_points2 = mask_loc.unsqueeze(1).expand(batch_size, m, m, 2)

        # 计算点之间的欧氏距离
        A_spa = torch.norm(expanded_points1 - expanded_points2, dim=3)
        A_spa = F.softmax(A_spa, dim=2)
        # =====================================
        gcn1 = self.GCN_layer1(pooled_features, A_spa)
        gcn2 = self.GCN_layer2(gcn1, A_spa)

        gcn_pred = self.gcn_projector(gcn2)

        pred = (gcn_pred + cnn_pred) / 2

        return pred


class AAM_MT(nn.Module):
    """
    multi-task AAM
    """

    def __init__(self, mask_num=30, feat_num=64):
        super(AAM_MT, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.feat_num = feat_num
        self.mask_num = mask_num
        self.feature_extractor = FCN2(in_channel=mask_num, out_channel=feat_num, bias=True,
                                      groups=mask_num)
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Flatten()

        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)

        self.projector_reg = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(feat_num * mask_num + 2048, 1, bias=False),
            nn.Sigmoid()
        )
        self.projector_cls = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(feat_num * mask_num + 2048, 2, bias=False),
            nn.Softmax(dim=1)
        )
        # initial feature extractor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, masks, mask_loc):
        masked_img = imgs.unsqueeze(1) * masks.unsqueeze(2)  # 2,3,224,224 -> 2,30,3,224,224
        bs = masked_img.shape[0]
        masked_img = torch.mean(masked_img, dim=2, keepdim=False)

        feats = self.feature_extractor(masked_img)
        feats = feats.reshape(bs, self.mask_num, -1, feats.shape[2],
                              feats.shape[3])  # 2,30,64,28,28

        pooled_features = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
        # 将分割为mask_num组

        # ============计算掩码空间关系============

        batch_size, m, _ = mask_loc.size()

        # 计算点之间的欧氏距离
        # 首先扩展张量以进行广播计算
        expanded_points1 = mask_loc.unsqueeze(2).expand(batch_size, m, m, 2)
        expanded_points2 = mask_loc.unsqueeze(1).expand(batch_size, m, m, 2)

        # 计算点之间的欧氏距离
        A_spa = torch.norm(expanded_points1 - expanded_points2, dim=3)
        A_spa = F.softmax(A_spa, dim=2)
        # =====================================
        gcn1 = self.GCN_layer1(pooled_features, A_spa)
        gcn2 = self.GCN_layer2(gcn1, A_spa)

        gcn2 = gcn2.view(gcn2.shape[0], -1)

        cnn_pred = self.cnn(imgs)
        mul_feat = torch.cat([gcn2, cnn_pred], dim=1)

        reg_pred = self.projector_reg(mul_feat)
        cls_pred = self.projector_cls(mul_feat)

        return reg_pred, cls_pred


class AAM_conv(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM_conv, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.mask_num = mask_num
        self.feat_num = feat_num
        self.feature_extractor = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=feat_num)
        self.GCN_layer1 = GraphConvLayer(dim_feature=feat_num)
        self.GCN_layer2 = GraphConvLayer(dim_feature=feat_num)
        if out_class == 1:
            self.projector = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(feat_num * mask_num, feat_num, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(feat_num, out_class),
                nn.Sigmoid()
            )
        else:
            self.projector = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(feat_num * mask_num, feat_num, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(feat_num, out_class),
                nn.Softmax(dim=1)
            )

    def forward(self, imgs, masks, mask_loc):
        feats = self.feature_extractor.backbone(imgs)['out']

        feats = feats.unsqueeze(2).repeat(1, 1, self.mask_num, 1, 1)

        feat_list = []
        for feat in feats:
            feat_list.append(F.conv2d(masks, feat, padding=28 // 2))

        feats = torch.cat(feat_list, dim=1)
        print(feats.shape)
        # 计算全局平均池化
        pooled_features = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)

        batch_size, m, _ = mask_loc.size()

        # 计算点之间的欧氏距离
        # 首先扩展张量以进行广播计算
        expanded_points1 = mask_loc.unsqueeze(2).expand(batch_size, m, m, 2)
        expanded_points2 = mask_loc.unsqueeze(1).expand(batch_size, m, m, 2)

        # 计算点之间的欧氏距离
        A_spa = torch.norm(expanded_points1 - expanded_points2, dim=3)
        A_spa = F.softmax(A_spa, dim=2)

        gcn1 = self.GCN_layer1(pooled_features, A_spa)
        gcn2 = self.GCN_layer2(gcn1, A_spa)

        pred = self.projector(gcn2)

        return pred


class AAM_attn(nn.Module):
    def __init__(self, mask_num=30, feat_num=64, out_class=10):
        super(AAM_attn, self).__init__()
        # self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.mask_num = mask_num
        self.feat_num = feat_num
        self.feature_extractor = FCN(in_channel=3, out_channel=feat_num)
        self.mask_extractor = resconvblock(1, feat_num, bias=True)
        self.v = nn.Conv2d(feat_num, feat_num, kernel_size=1, stride=1, padding=0, bias=False)

        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024 * (224 / (2 ** 4)) ** 2, 2048, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, out_class),
            nn.Sigmoid()
        )

    def forward(self, imgs, masks, mask_loc):
        feats = self.feature_extractor(imgs)

        # 创建一个形状为[1, feature_size, 1, 1]的权重tensor，用来乘以每个通道的下标
        weights = torch.arange(1, self.feat_num + 1, dtype=masks.dtype, device=masks.device).view(1,
                                                                                                  self.feat_num,
                                                                                                  1, 1)

        # 使用torch.sum和torch.mul计算每个通道的加权和
        weighted_masks = torch.sum(torch.mul(masks, weights), dim=1, keepdim=True)
        weighted_masks = F.sigmoid(weighted_masks)

        weighted_masks = self.mask_extractor(weighted_masks)
        weighted_masks = F.interpolate(weighted_masks, size=feats.size()[2:], mode="bilinear", align_corners=False)

        attn = F.softmax(weighted_masks * feats / 32, dim=1)
        feats = self.v(feats * attn) + feats

        pred = self.predictor(feats)

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
