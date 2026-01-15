import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        num_f = 32
        self.conv1 = Conv(3, num_f, 3)
        self.conv2 = Conv(num_f, num_f, 3)
        self.conv3 = Conv(num_f, num_f, 3)
        self.conv4 = Conv(num_f, num_f, 3)
        self.conv5 = Conv(num_f * 2, num_f, 3)
        self.conv6 = Conv(num_f * 2, num_f, 3)
        self.conv7 = nn.Conv2d(num_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.conv5(torch.cat([x3, x4], 1))
        x6 = self.conv6(torch.cat([x2, x5], 1))

        x_r = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        r_list = torch.split(x_r, 3, dim=1)

        x = x + r_list[0] * (torch.pow(x, 2) - x)
        x = x + r_list[1] * (torch.pow(x, 2) - x)
        x = x + r_list[2] * (torch.pow(x, 2) - x)
        x = x + r_list[3] * (torch.pow(x, 2) - x)
        x = x + r_list[4] * (torch.pow(x, 2) - x)
        x = x + r_list[5] * (torch.pow(x, 2) - x)
        x = x + r_list[6] * (torch.pow(x, 2) - x)
        x = x + r_list[7] * (torch.pow(x, 2) - x)

        r = torch.cat(r_list, dim=1)
        return x, r
