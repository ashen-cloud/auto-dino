import torch
import torch.nn as nn
import torchvision

class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.lins = nn.Sequential(
            nn.Linear(16224, 8112),
            nn.ReLU(inplace=True),
            nn.Linear(8112, 4056),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Linear(4056, 3) # jump, duck, nothing
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout(x)

        # print('shape', x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.lins(x)

        x = self.classifier(x)

        return x
