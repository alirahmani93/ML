import torch
import torch.nn.functional as F
from torch import nn


#
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, flag_relu: bool = True, **kwargs):
#         super().__init__()
#
#         self.cn = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.flag_relu = flag_relu
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.cn(x)
#         x = self.bn(x)
#         if self.flag_relu:
#             x = F.relu(x)
#         return x
#
#
# class ResUnit(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, flag_skip: bool, *args, **kwargs):
#         super().__init__()
#         self.flag_skip = flag_skip
#         self.cn1 = BasicConv2d(in_channels, out_channels, flag_relu=True)
#         self.cn2 = BasicConv2d(in_channels, out_channels, flag_relu=False)
#
#     def forward(self, x):
#         out = self.cn1(x)
#         out = self.cn2(out)
#         if self.flag_skip:
#             x = self.cn2(x)
#         out += x
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, in_channels: int, num_classes):
#         super().__init__()
#         self.cn1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=(3, 3))
#         self.s1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=(1, 1))
#         self.res_64 = nn.Sequential(
#             ResUnit(in_channels, out_channels=64, flag_skip=False),
#             ResUnit(in_channels, out_channels=64, flag_skip=False),
#             ResUnit(in_channels, out_channels=64, flag_skip=False),
#         )
#         self.res_128 = nn.Sequential(
#             ResUnit(64, out_channels=128, flag_skip=True),
#             ResUnit(128, out_channels=128, flag_skip=False),
#             ResUnit(128, out_channels=128, flag_skip=False),
#             ResUnit(128, out_channels=128, flag_skip=False),
#         )
#         self.res_256 = nn.Sequential(
#             ResUnit(128, out_channels=256, flag_skip=True),
#             ResUnit(256, out_channels=256, flag_skip=False),
#             ResUnit(256, out_channels=256, flag_skip=False),
#             ResUnit(256, out_channels=256, flag_skip=False),
#             ResUnit(256, out_channels=256, flag_skip=False),
#             ResUnit(256, out_channels=256, flag_skip=False),
#         )
#         self.res_512 = nn.Sequential(
#             ResUnit(256, out_channels=512, flag_skip=True),
#             ResUnit(512, out_channels=512, flag_skip=False),
#             ResUnit(512, out_channels=512, flag_skip=False),
#         )
#         self.gavgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(1024, num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.cn1(x)
#         x = self.s1(x)
#         x = self.res_64(x)
#         x = self.res_128(x)
#         x = self.res_256(x)
#         x = self.res_512(x)
#         x = self.b2(x)
#         x = self.gavgpool(x)
#         x = self.fc(x)
#         return x
#
#
# model = ResNet(64, 100)
# x = torch.randn(size=(1, 64, 7, 7))
# print(model(x))
######################################
# perpxility AI

# import torch
# import torch.nn.functional as F
# from torch import nn
#
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, flag_relu: bool = True, **kwargs):
#         super(BasicConv2d, self).__init__()
#
#         self.cn = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.flag_relu = flag_relu
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.cn(x)
#         x = self.bn(x)
#         if self.flag_relu:
#             x = F.relu(x)
#         return x
#
#
# class ResUnit(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, flag_skip: bool, *args, **kwargs):
#         super().__init__()
#         self.flag_skip = flag_skip
#         self.cn1 = BasicConv2d(in_channels, out_channels, flag_relu=True)
#         self.cn2 = BasicConv2d(out_channels, out_channels, flag_relu=False)
#         # Add a convolutional layer to map the input tensor to the same number of output channels
#         if flag_skip:
#             self.cn_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         else:
#             self.cn_skip = None
#
#     def forward(self, x):
#         out = self.cn1(x)
#         out = self.cn2(out)
#         if self.flag_skip:
#             if self.cn_skip is not None:
#                 x = self.cn_skip(x)
#             out += x
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, in_channels: int, num_classes):
#         super(ResNet, self).__init__()
#         self.cn1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=(3, 3))
#         self.s1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=(1, 1))
#         self.res_64 = nn.Sequential(
#             ResUnit(64, 64, flag_skip=False),
#             ResUnit(64, 64, flag_skip=False),
#             ResUnit(64, 64, flag_skip=False),
#         )
#         self.res_128 = nn.Sequential(
#             ResUnit(64, 128, flag_skip=True),
#             ResUnit(128, 128, flag_skip=False),
#             ResUnit(128, 128, flag_skip=False),
#             ResUnit(128, 128, flag_skip=False),
#         )
#         self.res_256 = nn.Sequential(
#             ResUnit(128, 256, flag_skip=True),
#             ResUnit(256, 256, flag_skip=False),
#             ResUnit(256, 256, flag_skip=False),
#             ResUnit(256, 256, flag_skip=False),
#             ResUnit(256, 256, flag_skip=False),
#             ResUnit(256, 256, flag_skip=False),
#         )
#         self.res_512 = nn.Sequential(
#             ResUnit(256, 512, flag_skip=True),
#             ResUnit(512, 512, flag_skip=False),
#             ResUnit(512, 512, flag_skip=False),
#         )
#         self.gavgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.cn1(x)
#         x = self.s1(x)
#         x = self.res_64(x)
#         x = self.res_128(x)
#         x = self.res_256(x)
#         x = self.res_512(x)
#         x = self.gavgpool(x)
#         x = nn.Flatten(x)
#         x = self.fc(x)
#         return x
#
#
# model = ResNet(3, 100)  # Changed the first argument to 3, assuming it's the number of input channels
# x = torch.randn(size=(1, 3, 224, 224))  # Changed the input size to match the number of input channels
# print(model(x))

########
# Ashkan

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, bot_layer=False):
        super().__init__()
        self.bot_layer = bot_layer
        self.cn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2 if bot_layer else 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.cn2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.cn_b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.cn1(x)
        out = self.cn2(out)
        if self.bot_layer:
            x = self.cn_b1(x)
        out += x
        out = F.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, stride=2, kernel_size=7, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block = nn.ModuleList()
        blocks = [[64] * 3, [128] * 4, [256] * 6, [512] * 3]
        for index, block in enumerate(blocks):
            temp_module = nn.ModuleList()
            if not index:
                for i in block:
                    temp_module.append(ResUnit(i, i))
            else:
                temp_module.append(ResUnit(block[0] // 2, block[0], bot_layer=True))
                for i in block[1:]:
                    temp_module.append(ResUnit(i, i))
            self.res_block.append(temp_module)

        self.out_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1000),
        )

    def forward(self, x):
        x = self.input_block(x)
        for subblock in self.res_block:
            for layer in subblock:
                x = layer(x)
        x = self.out_block(x)
        return x


# model = ResUnit(64, 128, bot_layer=True)
# # model = ResUnit(64, 64)
# X = torch.randn(size=(1, 64, 32, 32))
# print(model(X).size())
# print(sum([p.numel() for p in model.parameters()]))
model = ResNet34(3)
X = torch.randn(size=(2, 3, 32, 32))
print(model(X).size())
print(sum([p.numel() for p in model.parameters()]))
