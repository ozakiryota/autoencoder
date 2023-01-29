import torch


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        conv_unit_ch = 32

        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2000, 32 * conv_unit_ch, kernel_size=(3, 4), stride=1, padding=0),
            torch.nn.BatchNorm2d(32 * conv_unit_ch),
            torch.nn.ReLU(inplace=True),
            ## ch x 3 x 4
            
            torch.nn.ConvTranspose2d(32 * conv_unit_ch, 16 * conv_unit_ch, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(16 * conv_unit_ch),
            torch.nn.ReLU(inplace=True),
            ## ch x 6 x 8

            torch.nn.ConvTranspose2d(16 * conv_unit_ch, 8 * conv_unit_ch, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(8 * conv_unit_ch),
            torch.nn.ReLU(inplace=True),
            ## ch x 12 x 16

            torch.nn.ConvTranspose2d(8 * conv_unit_ch, 4 * conv_unit_ch, kernel_size=(4, 5), stride=1, padding=0),
            torch.nn.BatchNorm2d(4 * conv_unit_ch),
            torch.nn.ReLU(inplace=True),
            ## ch x 15 x 20

            torch.nn.ConvTranspose2d(4 * conv_unit_ch, 2 * conv_unit_ch, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(2 * conv_unit_ch),
            torch.nn.ReLU(inplace=True),
            ## ch x 30 x 40

            torch.nn.ConvTranspose2d(2 * conv_unit_ch, conv_unit_ch, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(conv_unit_ch),
            torch.nn.ReLU(inplace=True),
            ## ch x 60 x 80

            torch.nn.ConvTranspose2d(conv_unit_ch, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
            ## ch x 120 x 160
        )

    def forward(self, inputs):
        outputs = inputs.view(inputs.size(0), -1, 1, 1)
        outputs = self.deconv(outputs)
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
    gen_net = Decoder().to(device)
    gen_net.train()
    outputs = gen_net(inputs)
    ## debug
    print(gen_net)
    print("outputs.size() =", outputs.size())
    img_numpy = np.clip(outputs[0].cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
    plt.imshow(img_numpy)
    plt.show()


if __name__ == '__main__':
    test()