import os
import torch
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ZFNet, self).__init__()

        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Local contrast norm.이 있어야 하는데 파이토치에는 해당 클래스가 없는 듯? LocalResponseNorm 과는 다른 건가?
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True), # return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later

            # layer 2
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True),

            # layer 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # layer 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # layer 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
        )
        self.classifier = nn.Sequential(
            # layer 6
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # layer 7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
        )

        self.feature_outputs = [0]*len(self.features)
        self.switch_indices = dict()
        self.sizes = dict()

    def forward(self, x):

        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                self.feature_outputs[i] = x
                self.switch_indices[i] = indices
            else:
                x = layer(x)
                self.feature_outputs[i] = x

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet(**kwargs):
    model = ZFNet(**kwargs)
    return model