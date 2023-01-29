from PIL import Image
import torch


class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path_list_list, data_transformer, is_train, given_label=0):
        self.file_path_list_list = file_path_list_list
        self.data_transformer = data_transformer
        self.is_train = is_train
        self.given_label = given_label

    def __len__(self):
        return len(self.file_path_list_list)

    def __getitem__(self, index):
        img_path = self.file_path_list_list[index][0]
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = self.data_transformer(img_pil, self.is_train)
        label = self.given_label
        return img_tensor, label


def test():
    import os

    from filelist_maker import makeFileList
    from data_transformer import DataTransformer

    dir_path_list = [os.path.join(os.environ['HOME'], 'dataset/airsim/sample')]
    csv_name = 'file_list.csv'
    target_col_list = [1, 2]
    file_path_list_list = makeFileList(dir_path_list, csv_name, target_col_list)

    resize = (240, 320)
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    data_transformer = DataTransformer(resize, mean, std)

    dataset = AnomalyDataset(file_path_list_list, data_transformer, is_train=True)

    print("dataset.__len__() =", dataset.__len__())
    index = 0
    print("index", index, ":", dataset.__getitem__(index)[0].size())
    print("index", index, ":", dataset.__getitem__(index)[1])


if __name__ == '__main__':
    test()