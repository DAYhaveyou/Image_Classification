# coding=utf8
import torch
import numpy as np
import os
import time
from DataLoader.data_loader import DogCatImgLoader, train_transform, test_transform
from torch.utils.data import DataLoader

from Util.util import load_config_file, accuracy, AverageMeter, CheckpointMeter, _loger, check_output_path, load_model

# USED GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')


hook_dict = dict()


def farward_hook(module, inp, outp):
    hook_dict['input'] = inp
    hook_dict['output'] = outp


def save_feature2_npy(data, label, arg_paras, fea_path):
    # data_name = arg_paras['experiment_name']+"_feature"+".npy"
    # label_name = arg_paras['experiment_name'] + "_label" + ".npy"
    data_name = arg_paras['experiment_name'] + "_feature"
    label_name = arg_paras['experiment_name'] + "_label"
    np.save(os.path.join(fea_path, data_name), data)
    np.save(os.path.join(fea_path, label_name), label)


def load_feature2_npy(arg_paras, fea_path):
    data_name = arg_paras['experiment_name']+"_feature"+".npy"
    label_name = arg_paras['experiment_name'] + "_label" + ".npy"
    data_p = os.path.join(fea_path, data_name)
    label_p = os.path.join(fea_path, label_name)
    if os.path.exists(data_p) and os.path.exists(label_p):
        data = np.load(data_p)
        label = np.load(label_p)
        return data, label
    else:
        exit(0)
        return None


def extract_feature(arg_paras, data_loader):
    # val_data_loader = DogCatImgLoader(arg_paras['file_path']['val'], arg_paras['annotation_path']["val"],
    #                                test_transform(arg_paras['crop_size'], arg_paras['scale_size']))
    # val_loader = DataLoader(dataset=val_data_loader, batch_size=1, pin_memory=True, shuffle=False)

    model = load_model(arg_paras)

    feature_ar = []
    label_ar = []

    if arg_paras['GPU']:
        model = model.cuda()

    '''modify your feature layer here'''
    model.base_model.register_forward_hook(farward_hook)

    # layers_dict = {'base_model.layer4': 'fea'}
    # fea_extractor = FeatureExtractor(model, layers_dict)
    model.eval()
    epoch_steps = len(data_loader)
    iter_loader = iter(data_loader)
    for i in range(epoch_steps):
        value = next(iter_loader)
        data = value['data']
        labels = value['label']
        img_dir = value['img_dir']

        if arg_paras['GPU']:
            input = data.cuda()
        else:
            input = data

        output = model(input)
        result = hook_dict['output']
        feature_ar.append(result.cpu().detach().numpy()[0])
        label_ar.append(int(labels[0].numpy()))

        if i % arg_paras['train_print_freq'] == 0:
            print("PROCESSED IMG, ", img_dir[0])

    feature_path = check_output_path(arg_paras['output_path'], atx='Features')
    save_feature2_npy(feature_ar, label_ar, arg_paras, feature_path)


if __name__ == '__main__':

    pass

