import torch

import sys
sys.path.append('../')
from mod.prime_factorize import primeFactorize


class RgbSegDecoder(torch.nn.Module):
    def __init__(self, img_height, img_width, z_dim, conv_unit_ch, deconv_unit_ch):
        super(RgbSegDecoder, self).__init__()

        height_factor_list = primeFactorize(img_height, is_ascending_order=False)
        width_factor_list = primeFactorize(img_width, is_ascending_order=False)
        while len(height_factor_list) != len(width_factor_list):
            if len(height_factor_list) < len(width_factor_list):
                height_factor_list.append(1)
            else:
                width_factor_list.append(1)
        
        self.deconv_list = []
        for i, (height_factor, width_factor) in enumerate(zip(height_factor_list, width_factor_list)):
            concat_ch_dim = (len(height_factor_list) - i) * conv_unit_ch
            in_ch_dim = (len(height_factor_list) - i) * deconv_unit_ch + concat_ch_dim
            out_ch_dim = (len(height_factor_list) - i - 1) * deconv_unit_ch
            if i == 0:
                tmp_deconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(z_dim + concat_ch_dim, out_ch_dim, kernel_size=(height_factor, width_factor), stride=1, padding=0),
                    torch.nn.BatchNorm2d(out_ch_dim),
                    torch.nn.ReLU(inplace=True)
                )
            elif i == len(height_factor_list) - 1:
                tmp_deconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_ch_dim, 3, kernel_size=(height_factor + 2, width_factor + 2), stride=(height_factor, width_factor), padding=1),
                    torch.nn.Tanh()
                )
            else:
                tmp_deconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_ch_dim, out_ch_dim, kernel_size=(height_factor + 2, width_factor + 2), stride=(height_factor, width_factor), padding=1),
                    torch.nn.BatchNorm2d(out_ch_dim),
                    torch.nn.ReLU(inplace=True)
                )
            self.deconv_list.append(tmp_deconv)
        self.deconv_list = torch.nn.ModuleList(self.deconv_list)

    def forward(self, rgb_feature, seg_feature_list):
        rgb_feature = rgb_feature.view(rgb_feature.size(0), -1, 1, 1)
        seg_feature_list.reverse()
        for i, (deconv, seg_feature) in enumerate(zip(self.deconv_list, seg_feature_list)):
            if i == 0:
                inputs = torch.cat((rgb_feature, seg_feature), dim=1)
                outputs = deconv(inputs)
            else:
                inputs = torch.cat((outputs, seg_feature), dim=1)
                outputs = deconv(inputs)
        return outputs


def test():
    import numpy as np
    import matplotlib.pyplot as plt

    from seg_encoder import SegEncoder
    
    ## device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## data
    batch_size = 10
    z_dim = 5000
    img_ch = 10
    img_height = 120
    img_width = 160
    conv_unit_ch = 16
    rgb_feature = torch.randn(batch_size, z_dim).to(device)
    seg_enc_net = SegEncoder(img_ch, img_height, img_width, z_dim, conv_unit_ch).to(device)
    seg_feature_list = seg_enc_net(torch.randn(batch_size, img_ch, img_height, img_width).to(device))
    ## decode
    deconv_unit_ch = 32
    dec_net = RgbSegDecoder(img_height, img_width, z_dim, conv_unit_ch, deconv_unit_ch).to(device)
    dec_net.train()
    outputs = dec_net(rgb_feature, seg_feature_list)
    ## debug
    print(dec_net)
    print("outputs.size() =", outputs.size())
    img_numpy = np.clip(outputs[0].cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
    plt.imshow(img_numpy)
    plt.show()


if __name__ == '__main__':
    test()