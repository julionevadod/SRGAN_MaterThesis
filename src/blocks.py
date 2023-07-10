from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride = 1, padding = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels, momentum = 0.5),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels, momentum = 0.5)
        )
    def forward(self,x):
        input = x
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        final_output = torch.add(input,out2)
        return final_output