import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import torch

import sys
sys.path.append('../')
from mod.filelist_maker import makeFileList
from mod.data_transformer import DataTransformer
from mod.dataset import AnomalyDataset
from network.encoder import Encoder
from network.decoder import Decoder
from mod.diff_image_creator import getDiffImage


class Evaluator:
    def __init__(self):
        self.args = self.setArgument().parse_args()
        self.checkArgument()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.dataset = self.getDataset()
        self.enc_net, self.dec_net = self.getNetwork()
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
    
    def setArgument(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--device', default='cuda:0')
        arg_parser.add_argument('--dataset_dirs', nargs='+', required=True)
        arg_parser.add_argument('--csv_name', default='file_list.csv')
        arg_parser.add_argument('--csv_target_col', nargs='+', type=int)
        arg_parser.add_argument('--img_height', type=int, default=240)
        arg_parser.add_argument('--img_width', type=int, default=320)
        arg_parser.add_argument('--z_dim', type=int, default=100)
        arg_parser.add_argument('--conv_unit_ch', type=int, default=32)
        arg_parser.add_argument('--load_weights_dir', default='../../weights')
        arg_parser.add_argument('--save_fig_dir', default='../../fig')
        arg_parser.add_argument('--flag_show_reconstracted_images', action='store_true')
        arg_parser.add_argument('--show_h', type=int, default=5)
        arg_parser.add_argument('--show_w', type=int, default=10)
        return arg_parser

    def checkArgument(self):
        device_list = ['cpu', 'cuda'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
        if self.args.device not in device_list:
            self.args.device = 'cuda:0'

    def getDataset(self):
        ## file list
        dir_path_list = self.args.dataset_dirs
        file_path_list_list = makeFileList(dir_path_list, self.args.csv_name, self.args.csv_target_col)
        ## data transformer
        mean = ([0.5, 0.5, 0.5])
        std = ([0.5, 0.5, 0.5])
        self.data_transformer = DataTransformer((self.args.img_height, self.args.img_width), mean, std)
        ## dataset
        dataset = AnomalyDataset(file_path_list_list, self.data_transformer, is_train=False)
        return dataset

    def getNetwork(self):
        enc_net = Encoder(self.args.img_height, self.args.img_width, self.args.z_dim, is_train=False)
        dec_net = Decoder(self.args.img_height, self.args.img_width, self.args.z_dim, self.args.conv_unit_ch)

        enc_weights_path = os.path.join(self.args.load_weights_dir, 'encoder.pth')
        dec_weights_path = os.path.join(self.args.load_weights_dir, 'decoder.pth')
        if self.device == torch.device('cpu'):
            loaded_enc_weights = torch.load(enc_weights_path, map_location={"cuda:0": "cpu"})
            print("load [GPU -> CPU]:", enc_weights_path)
            loaded_dec_weights = torch.load(dec_weights_path, map_location={"cuda:0": "cpu"})
            print("load [GPU -> CPU]:", dec_weights_path)
        else:
            loaded_enc_weights = torch.load(enc_weights_path)
            print("load [GPU -> GPU]:", enc_weights_path)
            loaded_dec_weights = torch.load(dec_weights_path)
            print("load [GPU -> GPU]:", dec_weights_path)
        enc_net.load_state_dict(loaded_enc_weights)
        dec_net.load_state_dict(loaded_dec_weights)

        enc_net.to(self.device)
        dec_net.to(self.device)

        enc_net.eval()
        dec_net.eval()

        return enc_net, dec_net

    def evaluate(self):
        images_list = []
        label_list = []
        score_list = []

        for i in tqdm(range(len(self.dataset))):
            inputs = self.dataset.__getitem__(i)[0].unsqueeze(0).to(self.device)
            label = self.dataset.__getitem__(i)[1]

            with torch.set_grad_enabled(False):
                z = self.enc_net(inputs)
                outputs = self.dec_net(z)
                inputs = self.data_transformer.inverseNormalizedImage(inputs)
                outputs = self.data_transformer.inverseNormalizedImage(outputs)
                anomaly_score = self.l1_criterion(inputs, outputs).item()

            images_list.append([inputs.squeeze(0).cpu().detach().numpy(), outputs.squeeze(0).cpu().detach().numpy()])
            label_list.append(label)
            score_list.append(anomaly_score)

        print("# of anomaly samples:", label_list.count(1), "/", len(label_list))

        random_indicies = list(range(len(score_list)))
        random.shuffle(random_indicies)
        self.saveSortedImages(images_list, label_list, random_indicies, self.args.show_h, self.args.show_w,
            'random' + str(self.args.show_h * self.args.show_w) + '.png')
        sorted_indicies = np.argsort(score_list)
        self.saveSortedImages(images_list, label_list, sorted_indicies, self.args.show_h, self.args.show_w,
            'top' + str(self.args.show_h * self.args.show_w) + '_smallest_score.png')
        self.saveSortedImages(images_list, label_list, sorted_indicies[::-1], self.args.show_h, self.args.show_w,
            'top' + str(self.args.show_h * self.args.show_w) + '_largest_score.png')
        self.saveScoreGraph(score_list, label_list)
        plt.show()

    def saveSortedImages(self, images_list, label_list, indicies, h, w, save_name):
        num_shown = h * w

        if self.args.flag_show_reconstracted_images:
            h = 3 * h

        scale = 3
        plt.figure(figsize=(scale * h, scale * w))

        for i, index in enumerate(indicies):
            subplot_index = i + 1
            if subplot_index > num_shown:
                break
            
            real_image_np = images_list[index][0]
            real_image_np = np.clip(real_image_np.transpose((1, 2, 0)), 0, 1)

            if self.args.flag_show_reconstracted_images:
                subplot_index = 3 * w * (i // w) + (i % w) + 1

                reconstracted_image_np = images_list[index][1]
                reconstracted_image_np = np.clip(reconstracted_image_np.transpose((1, 2, 0)), 0, 1)
                plt.subplot(h, w, subplot_index + w, xlabel="net(x" + str(i + 1) + ")")
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                plt.imshow(reconstracted_image_np)

                diff_image_np = getDiffImage(real_image_np, reconstracted_image_np, max_value=1)
                plt.subplot(h, w, subplot_index + 2 * w, xlabel="x" + str(i + 1) + "-net(x" + str(i + 1) + ")")
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                plt.imshow(diff_image_np)

            sub_title = "anormal" if label_list[index] else ""
            plt.subplot(h, w, subplot_index, xlabel="x" + str(i + 1), ylabel=sub_title)
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            plt.imshow(real_image_np)

        plt.tight_layout()
        os.makedirs(self.args.save_fig_dir, exist_ok=True)
        plt.savefig(os.path.join(self.args.save_fig_dir, save_name))

    def saveScoreGraph(self, score_list, label_list):
        plt.figure()
        plt.xlabel("Anomaly score")
        plt.ylabel("Anomaly label")
        plt.scatter(score_list, label_list)
        plt.yticks([0.0, 1.0], [False, True])
        plt.savefig(os.path.join(self.args.save_fig_dir, 'anomaly_score'))


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.evaluate()