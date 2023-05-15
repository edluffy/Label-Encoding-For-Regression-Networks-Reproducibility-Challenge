import torch
import torch.nn as nn
import math

class ResNet(nn.Module):
    def __init__(self, block, layers, n_bits, code):
        super().__init__()

        self.code_bits = {
            'u': [200, 200],
            'j': [100, 100],
            'b1jdj': [51, 51],
            'b2jdj': [27, 27],
            'hex16': [16, 16]
        }

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._create_layer(block, 64, layers[0])
        self.layer2 = self._create_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._create_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._create_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.fc_angles_yaw = nn.Linear(512 * block.expansion, n_bits)
        self.fc_angles_pitch = nn.Linear(512 * block.expansion, n_bits)
        self.fc_angles_roll = nn.Linear(512 * block.expansion, n_bits)

        self.yawm = nn.Linear(n_bits, self.code_bits[code][0])
        self.pitchm = nn.Linear(n_bits, self.code_bits[code][0])
        self.rollm = nn.Linear(n_bits, self.code_bits[code][0])

        self._initialize_weights()

    def _create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        yaw = self.yawm(self.fc_angles_yaw(x))
        pitch = self.pitchm(self.fc_angles_pitch(x))
        roll = self.rollm(self.fc_angles_roll(x))

        return yaw, pitch, roll, 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
