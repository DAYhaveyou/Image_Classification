import torch.utils.data as data
## self define file
import time
from DataLoader.transform import *
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import os
from numpy.random import randint
import json

'''
#1)  DEFINED IMAGE TRANSFORM 
'''

def train_transform(crop_size):
    trans_train = torchvision.transforms.Compose([
        GroupMultiScaleCrop(int(crop_size), [1, .875, .75, .66]),
        GroupRandomHorizontalFlip(is_flow=False),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    return trans_train


def test_transform(crop_size, scale_size):
    trans_test = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(int(crop_size)),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    return trans_test


'''
#2)  DEFINED DATA LOADER 
'''

class DogCatImgLoader(data.Dataset):
    def __init__(self, root_path, list_file, transform=None):

        self.root_path = root_path  # original images path
        self.list_file = list_file  # corresponding label dir of data
        self.transform = transform  # image transform function
        self.data_list = self.extract_annotation()  # data list: [image_name, image_id]

    def extract_annotation(self):
        f = open(self.list_file, 'r')
        value_arr = json.load(f)
        ret_arr = []
        for vs in value_arr:
            ret_arr.append(vs)
        # demo ret_arr: [[image_name, image_id], ...
        return ret_arr

    def _load_image(self, img_dir):
        '''
        :param img_dir: original image path
        :return: digital data of orignal image
        '''
        img_dir = os.path.join(self.root_path, img_dir)
        try:
            return [Image.open(img_dir).convert('RGB')]
        except Exception:
            print('error loading image:', img_dir)
            return [Image.open(img_dir).convert('RGB')]

    def __getitem__(self, index):
        '''
        describe: default function of data-loader class
        :param index:
        :return: training data(process data), training label(label)
        '''
        record = self.data_list[index]
        img_dir = record[0]
        label = record[1]
        img_val = self._load_image(img_dir)  # digital data
        process_data = self.transform(img_val)  # data augmentation and normalization

        return index, process_data, label

    def __len__(self):
        '''
        Description: default function of data-loader class
        :return: the number of samples in data list
        '''
        return len(self.data_list)

'''
#3) TEST DEMOS
'''

def test_dataloader():
    root_p = 'D:/CODES_WATL/Codes_For_Image_Classifing/data/DogCat/train'
    file_dir = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Labels/dc_val.json'
    file_dir1 = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Labels/dc_train.json'
    transform = train_transform(crop_size=224)
    data_loader = DogCatImgLoader(root_p, file_dir1, transform)

    loader = DataLoader(dataset=data_loader, batch_size=16, shuffle=True, num_workers=8)
    start_time = time.time()
    for epoch in range(1):
        for i, value in enumerate(loader):
            idnexs, data, labels = value
            print(data.shape)
            print(labels)

    print('cost time: ', time.time()-start_time)


def test_dataloader1():
    root_p = 'D:/CODES_WATL/Codes_For_Image_Classifing/data/DogCat/train'
    file_dir = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Labels/dc_val.json'
    file_dir1 = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Labels/dc_train.json'
    transform = train_transform(crop_size=224)
    data_loader = DogCatImgLoader(root_p, file_dir1, transform)

    loader = DataLoader(dataset=data_loader, batch_size=16, shuffle=True, num_workers=8)
    epoch_steps = len(loader)
    iter_loader = iter(loader)
    for i in range(epoch_steps):
        _, data, labels = next(iter_loader)
        print(data.shape)
        print(labels)

def load_config_file(config_file):
    all_params = json.load(open(config_file))
    print(all_params)
    return all_params


def test1():
    config_file = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Config/config.json'
    arg_paras = load_config_file(config_file)

    file_dir = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Labels/dc_val.json'
    file_dir1 = 'D:/CODES_WATL/Codes_For_Image_Classifing/senet.pytorch-master/Labels/dc_train.json'

    if os.path.exists(arg_paras['file_path']['train']):
        print("yes")
        print("yes")


    train_data_loader = DogCatImgLoader(arg_paras['file_path']['train'], file_dir1)
    val_data_loader = DogCatImgLoader(arg_paras['file_path']['val'], file_dir)
    train_loader = DataLoader(dataset=train_data_loader, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data_loader, batch_size=1, pin_memory=True, shuffle=False)

    epoch_steps = len(train_loader)
    iter_loader = iter(train_loader)
    for i in range(epoch_steps):
        _, data, labels = next(iter_loader)
        print(data.shape)
        print(labels)

if __name__ == '__main__':
    # test_dataloader()
    # test_dataloader1()
    test1()




