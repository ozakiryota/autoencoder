import torch

import sys
sys.path.append('../')
from mod.prime_factorize import primeFactorize


class Decoder(torch.nn.Module):
    def __init__(self, img_height, img_width, z_dim):
        super(Decoder, self).__init__()

        height_factor_list = primeFactorize(img_height, is_ascending_order=False)
        width_factor_list = primeFactorize(img_width, is_ascending_order=False)
        while len(height_factor_list) != len(width_factor_list):
            if len(height_factor_list) < len(width_factor_list):
                height_factor_list.append(1)
            else:
                width_factor_list.append(1)
        
        conv_unit_ch = 32
        deconv_list = []
        for i, (height_factor, width_factor) in enumerate(zip(height_factor_list, width_factor_list)):
            in_ch_dim = 2 ** (len(height_factor_list) - i - 1)
            out_ch_dim = in_ch_dim // 2
            if i == len(height_factor_list) - 1:
                deconv_list.append(torch.nn.ConvTranspose2d(in_ch_dim * conv_unit_ch, 3, kernel_size=(height_factor + 2, width_factor + 2), stride=(height_factor, width_factor), padding=1))
                deconv_list.append(torch.nn.Tanh())
            else:
                if i == 0:
                    deconv_list.append(torch.nn.ConvTranspose2d(z_dim, out_ch_dim * conv_unit_ch, kernel_size=(height_factor, width_factor), stride=1, padding=0))
                else:
                    deconv_list.append(torch.nn.ConvTranspose2d(in_ch_dim * conv_unit_ch, out_ch_dim * conv_unit_ch, kernel_size=(height_factor + 2, width_factor + 2), stride=(height_factor, width_factor), padding=1))
                deconv_list.append(torch.nn.BatchNorm2d(out_ch_dim * conv_unit_ch))
                deconv_list.append(torch.nn.ReLU(inplace=True))
        self.deconv = torch.nn.Sequential(*deconv_list)

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1, 1, 1)
        outputs = self.deconv(inputs)
        return outputs


def test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    ## device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## data
    batch_size = 10
    z_dim = 5000
    inputs = torch.randn(batch_size, z_dim).to(device)
    ## decode
    img_height = 120
    img_width = 160
    dec_net = Decoder(img_height, img_width, z_dim).to(device)
    dec_net.train()
    outputs = dec_net(inputs)
    ## debug
    print(dec_net)
    print("outputs.size() =", outputs.size())
    img_numpy = np.clip(outputs[0].cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
    plt.imshow(img_numpy)
    plt.show()


if __name__ == '__main__':
    test()