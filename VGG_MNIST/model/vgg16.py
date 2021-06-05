import torch
import torch.nn as nn

import math

class VGG16(nn.Module):

    def __init__(self, _input_channel, num_class):
        super().__init__()

        # 모델 구현
        self.conv = nn.Sequential(
            # vgg의 커널사이즈는 3x3으로 고정
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), nn.LeakyReLU(0.2), # 224x224x3 -> 224x224x64 / padding=1, stride=1, kernel=3이면 인풋 크기와 같음
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224x224x64 -> 112x112x64

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112x128 -> 56x56x128

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56x256 -> 28x28x256

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28x512 -> 14x14x512

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14x512 -> 7x7x512
        )

        self.fcLayer = nn.Sequential(
            nn.Linear(7*7*512,4096), # 7x7x512를 fc하여 1x1x4096으로
            nn.LeakyReLU(0.2),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, num_class),
            #nn.Softmax(dim=1),
        )

    def forward(self, x):
        # forward 구현
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fcLayer(x)
        return x