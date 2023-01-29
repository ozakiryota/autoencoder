from PIL import ImageOps
import random

from torchvision import transforms


class DataTransformer():
    def __init__(self, resize, mean, std):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.img_transformer = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img_pil, is_train):
        if is_train:
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                img_pil = ImageOps.mirror(img_pil)
        img_tensor = self.img_transformer(img_pil)
        return img_tensor


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