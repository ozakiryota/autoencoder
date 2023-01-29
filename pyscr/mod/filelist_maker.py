import csv
import os
from PIL import Image


def makeFileList(dir_path_list, csv_name, target_col_list=None):
    file_path_list_list = []
    for dir_path in dir_path_list:
        csv_path = os.path.join(dir_path, csv_name)
        with open(csv_path) as file_name_list_csv:
            file_name_list_list = csv.reader(file_name_list_csv)
            for file_name_list in file_name_list_list:
                file_path_list = [os.path.join(dir_path, file_name) for file_name in file_name_list]
                if target_col_list != None:
                    file_path_list = [file_path_list[col] for col in target_col_list]
                file_path_list_list.append(file_path_list)
    return file_path_list_list


def test():
    dir_path_list = [os.path.join(os.environ['HOME'], 'dataset/airsim/sample')]
    csv_name = 'file_list.csv'
    target_col_list = [1, 2]
    file_path_list_list = makeFileList(dir_path_list, csv_name, target_col_list)

    print("len(file_path_list_list) = ", len(file_path_list_list))
    # print(file_path_list_list)
    if len(file_path_list_list) > 0:
        print("file_path_list_list[0] =", file_path_list_list[0])


if __name__ == '__main__':
    test()