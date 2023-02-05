import torch

import sys
sys.path.append('../')
from mod.prime_factorize import primeFactorize


class SegEncoder(torch.nn.Module):
    def __init__(self, img_ch, img_height, img_width, z_dim, conv_unit_ch):
        super(SegEncoder, self).__init__()

        height_factor_list = primeFactorize(img_height, is_ascending_order=True)
        width_factor_list = primeFactorize(img_width, is_ascending_order=True)
        while len(height_factor_list) != len(width_factor_list):
            if len(height_factor_list) < len(width_factor_list):
                height_factor_list.insert(0, 1)
            else:
                width_factor_list.insert(0, 1)
        
        self.conv_list = []
        for i, (height_factor, width_factor) in enumerate(zip(height_factor_list, width_factor_list)):
            in_ch_dim = i * conv_unit_ch
            out_ch_dim = (i + 1) * conv_unit_ch
            if i == 0:
                tmp_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(img_ch, out_ch_dim, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=(height_factor, width_factor), stride=(height_factor, width_factor), padding=0, dilation=1, ceil_mode=False)
                )
            else:
                tmp_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=(height_factor, width_factor), stride=(height_factor, width_factor), padding=0, dilation=1, ceil_mode=False)
                )
            self.conv_list.append(tmp_conv)
        self.conv_list = torch.nn.ModuleList(self.conv_list)

    def forward(self, x):
        outputs_list = []
        for conv in self.conv_list:
            x = conv(x)
            outputs_list.append(x)
        return outputs_list


def test():
    ## device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## data
    batch_size = 10
    img_ch = 10
    img_height = 120
    img_width = 160
    inputs = torch.randn(batch_size, img_ch, img_height, img_width).to(device)
    ## decode
    z_dim = 5000
    enc_net = SegEncoder(img_ch, img_height, img_width, z_dim).to(device)
    enc_net.train()
    outputs_list = enc_net(inputs)
    ## debug
    print(enc_net)
    for i, outputs in enumerate(outputs_list):
        print("outputs_list[", i, "].size() =", outputs.size())


if __name__ == '__main__':
    test()