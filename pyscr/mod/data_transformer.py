from PIL import ImageOps
import random
import math

import torch
from torchvision import transforms


class DataTransformer():
    def __init__(self, resize, mean, std, rotation_range_deg=None):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.img_transformer = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.seg_img_transformer = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.PILToTensor()
        ])
        self.inv_img_transformer = transforms.Compose([
            transforms.Normalize([0, 0, 0], [1 / s for s in std]),
            transforms.Normalize([-m for m in mean], [1, 1, 1])
        ])
        self.rotation_range_deg = rotation_range_deg

    def __call__(self, img_pil, seg_img_pil=None, is_train=True):
        if is_train:
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                img_pil = ImageOps.mirror(img_pil)
                if seg_img_pil != None:
                    seg_img_pil = ImageOps.mirror(seg_img_pil)
            if self.rotation_range_deg != None and self.rotation_range_deg != 0:
                angle_deg = random.uniform(-self.rotation_range_deg, self.rotation_range_deg)
                img_pil = img_pil.rotate(angle_deg)
                if seg_img_pil != None:
                    seg_img_pil = seg_img_pil.rotate(angle_deg)
        img_tensor = self.img_transformer(img_pil)
        if seg_img_pil == None:
            return img_tensor
        else:
            seg_img_tensor = self.seg_img_transformer(seg_img_pil).squeeze(0)
            seg_img_tensor = torch.nn.functional.one_hot(seg_img_tensor.to(torch.int64), num_classes=256)
            seg_img_tensor = torch.permute(seg_img_tensor, (2, 0, 1)).to(torch.float32)
            return img_tensor, seg_img_tensor

    def inverseNormalizedImage(self, img_tensor):
        return self.inv_img_transformer(img_tensor)


def test():
    import os
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    from filelist_maker import makeFileList

    dir_path_list = [os.path.join(os.environ['HOME'], 'dataset/airsim/sample')]
    csv_name = 'file_list.csv'
    target_col_list = [1]
    file_path_list_list = makeFileList(dir_path_list, csv_name, target_col_list)
    img_pil = Image.open(file_path_list_list[0][0]).convert('RGB')

    resize = (240, 320)
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    data_transformer = DataTransformer(resize, mean, std)
    img_tensor = data_transformer(img_pil, is_train=True)
    
    print("img_tensor.size() =", img_tensor.size())
    img_trans_numpy = img_tensor.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
    img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
    plt.subplot(2, 1, 1)
    plt.imshow(img_pil)
    plt.subplot(2, 1, 2)
    plt.imshow(img_trans_numpy)
    plt.show()


if __name__ == '__main__':
    test()