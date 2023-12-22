import torch
import torchvision
import torch.nn as nn

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(self.base_model.fc.in_features, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x