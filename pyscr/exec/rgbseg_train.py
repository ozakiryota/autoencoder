import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from exec.train import Trainer
from mod.filelist_maker import makeFileList
from mod.data_transformer import DataTransformer
from mod.dataset import AnomalyDataset
from network.encoder import Encoder as RgbEncoder
from network.seg_encoder import SegEncoder
from network.rgbseg_decoder import RgbSegDecoder


class RgbSegTrainer(Trainer):
    def __init__(self):
        self.args = self.setArgument().parse_args()
        self.checkArgument()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        print("self.device =", self.device)
        self.dataloader = self.getDataLoader()
        self.rgb_enc_net, self.seg_enc_net, self.dec_net = self.getNetwork()
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
        self.rgb_enc_optimizer, self.seg_enc_optimizer, self.dec_optimizer = self.getOptimizer()
        self.info_str = self.getInfoStr()
        self.tb_writer = self.getWriter()

    def getNetwork(self):
        rgb_enc_net = RgbEncoder(self.args.img_height, self.args.img_width, self.args.z_dim)
        seg_enc_net = SegEncoder(256, self.args.img_height, self.args.img_width, self.args.z_dim)
        dec_net = RgbSegDecoder(self.args.img_height, self.args.img_width, self.args.z_dim)

        if self.args.load_weights_dir is not None:
            rgb_enc_weights_path = os.path.join(self.args.load_weights_dir, 'rgb_encoder.pth')
            seg_enc_weights_path = os.path.join(self.args.load_weights_dir, 'seg_encoder.pth')
            dec_weights_path = os.path.join(self.args.load_weights_dir, 'decoder.pth')
            if self.device == torch.device('cpu'):
                loaded_rgb_enc_weights = torch.load(rgb_enc_weights_path, map_location={"cuda:0": "cpu"})
                print("load [GPU -> CPU]:", rgb_enc_weights_path)
                loaded_seg_enc_weights = torch.load(seg_enc_weights_path, map_location={"cuda:0": "cpu"})
                print("load [GPU -> CPU]:", seg_enc_weights_path)
                loaded_dec_weights = torch.load(dec_weights_path, map_location={"cuda:0": "cpu"})
                print("load [GPU -> CPU]:", dec_weights_path)
            else:
                loaded_rgb_enc_weights = torch.load(rgb_enc_weights_path)
                print("load [GPU -> GPU]:", rgb_enc_weights_path)
                loaded_seg_enc_weights = torch.load(seg_enc_weights_path)
                print("load [GPU -> GPU]:", seg_enc_weights_path)
                loaded_dec_weights = torch.load(dec_weights_path)
                print("load [GPU -> GPU]:", dec_weights_path)
            rgb_enc_net.load_state_dict(loaded_rgb_enc_weights)
            seg_enc_net.load_state_dict(loaded_seg_enc_weights)
            dec_net.load_state_dict(loaded_dec_weights)

        rgb_enc_net.to(self.device)
        seg_enc_net.to(self.device)
        dec_net.to(self.device)
        if self.args.flag_use_multi_gpu:
            rgb_enc_net = torch.nn.DataParallel(rgb_enc_net)
            seg_enc_net = torch.nn.DataParallel(seg_enc_net)
            dec_net = torch.nn.DataParallel(dec_net)

        return rgb_enc_net, seg_enc_net, dec_net

    def getOptimizer(self):
        beta1, beta2 = 0.5, 0.999
        rgb_enc_optimizer = torch.optim.Adam(self.rgb_enc_net.parameters(), self.args.enc_lr, [beta1, beta2])
        seg_enc_optimizer = torch.optim.Adam(self.seg_enc_net.parameters(), self.args.enc_lr, [beta1, beta2])
        dec_optimizer = torch.optim.Adam(self.dec_net.parameters(), self.args.dec_lr, [beta1, beta2])
        return rgb_enc_optimizer, seg_enc_optimizer, dec_optimizer

    def getInfoStr(self):
        info_str = super(RgbSegTrainer, self).getInfoStr()
        info_str = 'rgbseg' + info_str
        return info_str

    def train(self):
        torch.backends.cudnn.benchmark = True
        
        loss_record = []
        start_clock = time.time()

        for epoch in range(self.args.num_epochs):
            epoch_start_clock = time.time()

            epoch_loss = 0.0

            print("-------------")
            print("epoch: {}/{}".format(epoch + 1, self.args.num_epochs))

            for rgb_inputs, seg_inputs, _ in self.dataloader:
                batch_size_in_loop = rgb_inputs.size(0)
                rgb_inputs = rgb_inputs.to(self.device)
                seg_inputs = seg_inputs.to(self.device)
                loss = self.optimize(rgb_inputs, seg_inputs)
                epoch_loss += batch_size_in_loop * loss.item()
            self.record(epoch, loss_record, epoch_loss)
            print("epoch time: {:.1f} sec".format(time.time() - epoch_start_clock))
            print("total time: {:.1f} min".format((time.time() - start_clock) / 60))

            if (epoch + 1) % self.args.save_weights_step == 0 or (epoch + 1) == self.args.num_epochs:
                self.saveWeights(epoch + 1)
        print("-------------")
        ## save
        self.tb_writer.close()
        self.saveLossGraph(loss_record)
        plt.show()

    def optimize(self, rgb_inputs, seg_inputs):
        self.rgb_enc_net.train()
        self.seg_enc_net.train()
        self.dec_net.train()

        rgb_feature = self.rgb_enc_net(rgb_inputs)
        seg_feature_list = self.seg_enc_net(seg_inputs)
        outputs = self.dec_net(rgb_feature, seg_feature_list)

        loss = self.l1_criterion(rgb_inputs, outputs)

        self.rgb_enc_optimizer.zero_grad()
        self.seg_enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss.backward()

        self.rgb_enc_optimizer.step()
        self.seg_enc_optimizer.step()
        self.dec_optimizer.step()

        return loss

    def saveWeights(self, epoch):
        save_weights_dir = os.path.join(self.args.save_weights_dir, self.info_str)
        insert_index = save_weights_dir.find('batch') + len('batch')
        save_weights_dir = save_weights_dir[:insert_index] + str(epoch) + save_weights_dir[insert_index + len(str(self.args.num_epochs)):]
        os.makedirs(save_weights_dir, exist_ok=True)
        save_rgb_enc_weights_path = os.path.join(save_weights_dir, 'rgb_encoder.pth')
        save_seg_enc_weights_path = os.path.join(save_weights_dir, 'seg_encoder.pth')
        save_dec_weights_path = os.path.join(save_weights_dir, 'decoder.pth')
        if self.args.flag_use_multi_gpu:
            torch.save(self.rgb_enc_net.module.state_dict(), save_rgb_enc_weights_path)
            torch.save(self.seg_enc_net.module.state_dict(), save_seg_enc_weights_path)
            torch.save(self.dec_net.module.state_dict(), save_dec_weights_path)
        else:
            torch.save(self.rgb_enc_net.state_dict(), save_rgb_enc_weights_path)
            torch.save(self.seg_enc_net.state_dict(), save_seg_enc_weights_path)
            torch.save(self.dec_net.state_dict(), save_dec_weights_path)
        print("save:", save_rgb_enc_weights_path)
        print("save:", save_seg_enc_weights_path)
        print("save:", save_dec_weights_path)


if __name__ == '__main__':
    trainer = RgbSegTrainer()
    trainer.train()