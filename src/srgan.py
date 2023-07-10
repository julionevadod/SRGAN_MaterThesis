from torch import nn
from .blocks import ResidualBlock
import torch

class Discriminator(nn.Module):
    def __init__(self, input_resolution):
        h = input_resolution[1]
        w = input_resolution[0]
        c = input_resolution[2]
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(c, 64, 3, 1, 1, bias = True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1, bias = True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1, bias = True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1, bias = True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1, bias = True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 2, 1, bias = True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 1, 1, bias = True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 2, 1, bias = True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(1),
            nn.Linear(512*8*8,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()

        )
    
    def forward(self,x):
        output = self.network(x)
        return output

class Generator(nn.Module):
    def __init__(self, input_resolution, B):
        h = input_resolution[1]
        w = input_resolution[0]
        c = input_resolution[2]
        super(Generator, self).__init__()
        self.input_network = nn.Sequential(
            nn.Conv2d( c, 64, 9, 1, 4, bias = True),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.residual_network = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
                for b in range(B)
            ]
        )
        self.output_network1 = nn.Sequential(
            nn.Conv2d( 64, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
        )
        self.output_network2 = nn.Sequential(
            nn.Conv2d( 64, 256, 3, 1, 1, bias = False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d( 64, 256, 3, 1, 1, bias = False),
            nn.PixelShuffle(2), #As many pixel shufflers as sqrt(r)
            nn.PReLU(),
            nn.Conv2d(64, c, 9, 1, 4, bias = False)
        )


    def forward(self, input):
        input_network_output = self.input_network(input)
        residual_network_output = self.residual_network(input_network_output)
        output_network1_output = self.output_network1(residual_network_output)
        output = self.output_network2(torch.add(input_network_output,output_network1_output))
        return output
