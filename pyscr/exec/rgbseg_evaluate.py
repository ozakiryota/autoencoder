import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import torch

import sys
sys.path.append('../')
from exec.evaluate import Evaluator
from mod.filelist_maker import makeFileList
from mod.data_transformer import DataTransformer
from mod.dataset import AnomalyDataset
from network.encoder import Encoder as RgbEncoder
from network.seg_encoder import SegEncoder
from network.rgbseg_decoder import RgbSegDecoder
from mod.diff_image_creator import getDiffImage


class RgbSegEvaluator(Evaluator):
    def __init__(self):
        self.args = self.setArgument().parse_args()
        self.checkArgument()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.dataset = self.getDataset()
        self.rgb_enc_net, self.seg_enc_net, self.dec_net = self.getNetwork()
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')

    def setArgument(self):
        arg_parser = super(RgbSegEvaluator, self).setArgument()
        arg_parser.add_argument('--num_classes', type=int, default=256)
        arg_parser.add_argument('--conv_unit_ch', type=int, default=32)
        return arg_parser

    def getNetwork(self):
        rgb_enc_net = RgbEncoder(self.args.img_height, self.args.img_width, self.args.z_dim, is_train=False)
        seg_enc_net = SegEncoder(self.args.num_classes, self.args.img_height, self.args.img_width, self.args.z_dim, self.args.conv_unit_ch)
        dec_net = RgbSegDecoder(self.args.img_height, self.args.img_width, self.args.z_dim, self.args.conv_unit_ch, self.args.deconv_unit_ch)

        rgb_enc_weights_path = os.path.join(self.args.load_exp_dir, 'rgb_encoder.pth')
        seg_enc_weights_path = os.path.join(self.args.load_exp_dir, 'seg_encoder.pth')
        dec_weights_path = os.path.join(self.args.load_exp_dir, 'decoder.pth')
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

        rgb_enc_net.eval()
        seg_enc_net.eval()
        dec_net.eval()

        return rgb_enc_net, seg_enc_net, dec_net

    def evaluate(self):
        images_list = []
        label_list = []
        score_list = []

        for i in tqdm(range(len(self.dataset))):
            rgb_inputs = self.dataset.__getitem__(i)[0].unsqueeze(0).to(self.device)
            seg_inputs = self.dataset.__getitem__(i)[1].unsqueeze(0).to(self.device)
            label = self.dataset.__getitem__(i)[2]

            with torch.set_grad_enabled(False):
                rgb_feature = self.rgb_enc_net(rgb_inputs)
                seg_feature_list = self.seg_enc_net(seg_inputs)
                outputs = self.dec_net(rgb_feature, seg_feature_list)
                rgb_inputs = self.data_transformer.inverseNormalizedImage(rgb_inputs)
                outputs = self.data_transformer.inverseNormalizedImage(outputs)
                anomaly_score = self.l1_criterion(rgb_inputs, outputs).item()

            images_list.append([rgb_inputs.squeeze(0).cpu().detach().numpy(), outputs.squeeze(0).cpu().detach().numpy()])
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


if __name__ == '__main__':
    evaluator = RgbSegEvaluator()
    evaluator.evaluate()