import torch
from torchvision import models


class Encoder(torch.nn.Module):
    def __init__(self, img_height, img_width, z_dim, is_train=True):
        super(Encoder, self).__init__()

        if is_train:
            self.conv = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            # self.conv = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        else:
            self.conv = models.vgg16().features
            # self.conv = models.vgg19().features
        dim_fc_in = 512 * (img_height // 32) * (img_width // 32)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(dim_fc_in, dim_fc_in + (dim_fc_in - z_dim) // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_fc_in + (dim_fc_in - z_dim) // 2, z_dim)
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)
        return outputs


def test():
    ## device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## data
    batch_size = 10
    img_height = 240
    img_width = 320
    inputs = torch.randn(batch_size, 3, img_height, img_width).to(device)
    ## encode
    z_dim = 1000
    enc_net = Encoder(img_height, img_width, z_dim).to(device)
    enc_net.train()
    outputs = enc_net(inputs)
    ## debug
    print(enc_net)
    print("outputs.size() =", outputs.size())


if __name__ == '__main__':
    test()