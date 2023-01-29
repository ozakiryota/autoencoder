import numpy as np


def getDiffImage(img0_np, img1_np, max_value=255):
    img0_np = 255 / max_value * img0_np
    img1_np = 255 / max_value * img1_np
    diff_img_np = np.abs(img0_np.astype(int) - img1_np.astype(int))
    diff_img_np = diff_img_np[:, :, 0] + diff_img_np[:, :, 1] + diff_img_np[:, :, 2]
    return diff_img_np


def test():
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    from filelist_maker import makeFileList

    dataset_dir_list = [os.path.join(os.environ['HOME'], 'dataset/airsim/sample')]
    csv_name = 'file_list.csv'
    append_col_list = [1]
    file_list_list = makeFileList(dataset_dir_list, csv_name, append_col_list)
    print(file_list_list[0][0])
    img0_np = np.array(Image.open(file_list_list[0][0]).convert('RGB'))
    img1_np = np.array(Image.open(file_list_list[1][0]).convert('RGB'))
    diff_img_np = getDiffImage(img0_np, img1_np)
    plt.imshow(diff_img_np)
    plt.show()


if __name__ == '__main__':
    test()