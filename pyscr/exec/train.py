import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from mod.filelist_maker import makeFileList
from mod.data_transformer import DataTransformer
from mod.dataset import AnomalyDataset
from network.encoder import Encoder
from network.decoder import Decoder


class Trainer:
    def __init__(self):
        self.args = self.setArgument().parse_args()
        self.checkArgument()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.dataloader = self.getDataLoader()
        self.enc_net, self.dec_net = self.getNetwork()
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
        self.enc_optimizer, self.dec_optimizer = self.getOptimizer()
        self.info_str = self.getInfoStr()
        self.tb_writer = self.getWriter()
    
    def setArgument(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--device', default='cuda:0')
        arg_parser.add_argument('--dataset_dirs', nargs='+', required=True)
        arg_parser.add_argument('--csv_name', default='file_list.csv')
        arg_parser.add_argument('--csv_target_col', nargs='+', type=int)
        arg_parser.add_argument('--img_height', type=int, default=240)
        arg_parser.add_argument('--img_width', type=int, default=320)
        arg_parser.add_argument('--batch_size', type=int, default=100)
        # arg_parser.add_argument('--z_dim', type=int, default=5000)
        # arg_parser.add_argument('--conv_unit_ch', type=int, default=32)
        # arg_parser.add_argument('--load_weights_dir')
        arg_parser.add_argument('--flag_use_multi_gpu', action='store_true')
        arg_parser.add_argument('--enc_lr', type=float, default=1e-5)
        arg_parser.add_argument('--dec_lr', type=float, default=1e-5)
        arg_parser.add_argument('--num_epochs', type=int, default=100)
        arg_parser.add_argument('--save_weights_step', type=int)
        arg_parser.add_argument('--save_weights_dir', default='../../weights')
        arg_parser.add_argument('--save_log_dir', default='../../log')
        arg_parser.add_argument('--save_fig_dir', default='../../fig')

        return arg_parser

    def checkArgument(self):
        device_list = ['cpu', 'cuda'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
        if self.args.device not in device_list:
            self.args.device = 'cuda:0'
        if self.args.save_weights_step == None:
            self.args.save_weights_step = self.args.num_epochs
        else:
            self.args.save_weights_step = min(self.args.save_weights_step, self.args.num_epochs)

    def getDataLoader(self):
        ## file list
        dir_path_list = self.args.dataset_dirs
        file_path_list_list = makeFileList(dir_path_list, self.args.csv_name, self.args.csv_target_col)
        ## data transformer
        mean = ([0.5, 0.5, 0.5])
        std = ([0.5, 0.5, 0.5])
        data_transformer = DataTransformer((self.args.img_height, self.args.img_width), mean, std)
        ## dataset
        dataset = AnomalyDataset(file_path_list_list, data_transformer, is_train=True)
        ## dataloader
        dataloader = torch.utils.data.DataLoader(dataset, self.args.batch_size, shuffle=True, drop_last=True)
        return dataloader

    def getNetwork(self):
        enc_net = Encoder()
        dec_net = Decoder()

        # if self.args.load_weights_dir is not None:
        #     gen_weights_path = os.path.join(self.args.load_weights_dir, 'generator.pth')
        #     dis_weights_path = os.path.join(self.args.load_weights_dir, 'discriminator.pth')
        #     enc_weights_path = os.path.join(self.args.load_weights_dir, 'encoder.pth')
        #     if self.device == torch.device('cpu'):
        #         loaded_gen_weights = torch.load(gen_weights_path, map_location={"cuda:0": "cpu"})
        #         print("load [GPU -> CPU]:", gen_weights_path)
        #         loaded_dis_weights = torch.load(dis_weights_path, map_location={"cuda:0": "cpu"})
        #         print("load [GPU -> CPU]:", dis_weights_path)
        #         loaded_enc_weights = torch.load(enc_weights_path, map_location={"cuda:0": "cpu"})
        #         print("load [GPU -> CPU]:", enc_weights_path)
        #     else:
        #         loaded_gen_weights = torch.load(gen_weights_path)
        #         print("load [GPU -> GPU]:", gen_weights_path)
        #         loaded_dis_weights = torch.load(dis_weights_path)
        #         print("load [GPU -> GPU]:", dis_weights_path)
        #         loaded_enc_weights = torch.load(enc_weights_path)
        #         print("load [GPU -> GPU]:", enc_weights_path)
        #     gen_net.load_state_dict(loaded_gen_weights)
        #     dis_net.load_state_dict(loaded_dis_weights)
        #     enc_net.load_state_dict(loaded_enc_weights)

        enc_net.to(self.device)
        dec_net.to(self.device)
        if self.args.flag_use_multi_gpu:
            enc_net = torch.nn.DataParallel(enc_net)
            dec_net = torch.nn.DataParallel(dec_net)

        return enc_net, dec_net

    def getOptimizer(self):
        beta1, beta2 = 0.5, 0.999
        enc_optimizer = torch.optim.Adam(self.enc_net.parameters(), self.args.enc_lr, [beta1, beta2])
        dec_optimizer = torch.optim.Adam(self.dec_net.parameters(), self.args.dec_lr, [beta1, beta2])
        return enc_optimizer, dec_optimizer

    def getInfoStr(self):
        info_str = str(self.args.img_height) + 'pixel' \
            + str(self.args.enc_lr) + 'lre' \
            + str(self.args.dec_lr) + 'lrd' \
            + str(len(self.dataloader.dataset)) + 'sample' \
            + str(self.args.batch_size) + 'batch' \
            + str(self.args.num_epochs) + 'epoch'
        # if self.args.load_weights_dir is not None:
        #     insert_index = info_str.find('epoch')
        #     info_str = info_str[:insert_index] + '+' + info_str[insert_index:]
        info_str = info_str.replace('-', '').replace('.', '')

        print("self.device =", self.device)
        print("info_str =", info_str)

        return info_str

    def getWriter(self):
        save_log_dir = os.path.join(self.args.save_log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + self.info_str)
        tb_writer = SummaryWriter(logdir=save_log_dir)
        print("save_log_dir =", save_log_dir)
        return tb_writer

    def train(self):
        # torch.backends.cudnn.benchmark = True
        
        loss_record = []
        start_clock = time.time()

        for epoch in range(self.args.num_epochs):
            epoch_start_clock = time.time()

            epoch_loss = 0.0

            print("-------------")
            print("epoch: {}/{}".format(epoch + 1, self.args.num_epochs))

            for inputs, _ in self.dataloader:
                batch_size_in_loop = inputs.size(0)
                inputs = inputs.to(self.device)
                loss = self.optimize(inputs)
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

    def optimize(self, inputs):
        self.enc_net.train()
        self.dec_net.train()

        z = self.enc_net(inputs)
        outputs = self.dec_net(z)

        loss = self.l1_criterion(inputs, outputs)

        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss.backward()

        self.enc_optimizer.step()
        self.dec_optimizer.step()

        return loss

    def record(self, epoch, loss_record, epoch_loss):
        num_data = len(self.dataloader.dataset)
        loss_record.append(epoch_loss / num_data)
        self.tb_writer.add_scalars('loss', {'train': loss_record[-1]}, epoch)
        print("loss: {:.4f}".format(loss_record[-1]))

    def saveWeights(self, epoch):
        save_weights_dir = os.path.join(self.args.save_weights_dir, self.info_str)
        insert_index = save_weights_dir.find('batch') + len('batch')
        save_weights_dir = save_weights_dir[:insert_index] + str(epoch) + save_weights_dir[insert_index + len(str(self.args.num_epochs)):]
        os.makedirs(save_weights_dir, exist_ok=True)
        save_dec_weights_path = os.path.join(save_weights_dir, 'decoder.pth')
        save_enc_weights_path = os.path.join(save_weights_dir, 'encoder.pth')
        if self.args.flag_use_multi_gpu:
            torch.save(self.enc_net.module.state_dict(), save_enc_weights_path)
            torch.save(self.dec_net.module.state_dict(), save_dec_weights_path)
        else:
            torch.save(self.enc_net.state_dict(), save_enc_weights_path)
            torch.save(self.dec_net.state_dict(), save_dec_weights_path)
        print("save:", save_enc_weights_path)
        print("save:", save_dec_weights_path)

    def saveLossGraph(self, loss_record):
        plt.figure()
        plt.plot(range(len(loss_record)), loss_record, label="train")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("loss=" + '{:.4f}'.format(loss_record[-1]))

        fig_save_path = os.path.join(self.args.save_fig_dir, self.info_str + '.jpg')
        plt.savefig(fig_save_path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()