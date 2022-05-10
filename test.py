# coding=utf8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import time
from DataLoader.data_loader import DogCatImgLoader, train_transform, test_transform
from torch.utils.data import DataLoader
from Util.util import load_config_file, accuracy, AverageMeter, record_negative_samples, _loger, check_output_path, load_model
from model import Model

# USED GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')


def validation(model, data_loader, logger, arg_paras, atx='val'):
    ''' validate the trained model '''

    negative_sample_arr = []

    top1 = AverageMeter()
    model.eval()
    epoch_steps = len(data_loader)
    iter_loader = iter(data_loader)
    for i in range(epoch_steps):
        value = next(iter_loader)
        data = value['data']
        labels = value['label']
        img_dir = value['img_dir']
        input = data.cuda()
        label = labels.cuda()
        output = model(input)

        pred = accuracy(output, label)
        top1.update(pred[0].item(), input.size(0))
        # negative_sample_arr.append([img_dir, int(labels[0].numpy())])
        if pred[0].item() < 100:
            negative_sample_arr.append([img_dir[0], int(labels[0].numpy())])

        if i % arg_paras['train_print_freq'] == 0:
            logger.info(('=> Step: [{0}/{1}],\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(i, epoch_steps, top1=top1)))

    if len(negative_sample_arr) != 0:
        record_negative_samples(negative_sample_arr, arg_paras)

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f}'.format(top1=top1)))
    return top1.avg


def main():
    # load predefined setting
    config_file = 'Config/config.json'
    arg_paras = load_config_file(config_file)

    test_data_loader = DogCatImgLoader(arg_paras['file_path']['test'], arg_paras['annotation_path']["test"],
                                      test_transform(arg_paras['crop_size'], arg_paras['scale_size']))
    test_loader = DataLoader(dataset=test_data_loader, batch_size=1, pin_memory=True, shuffle=False)

    #  1) check output dictionary //检查输出文件路径是否存在
    log_dir = check_output_path(arg_paras['output_path'], atx='Logs')
    logger = _loger(log_dir, arg_paras['experiment_name'])

    # 2) load model and set optimizer
    model = load_model(arg_paras)
    if arg_paras['GPU']:
        model = model.to(device)

    acc = validation(model, test_loader, logger, arg_paras, atx='test')

    logger.info(('=> The Best Accuracy\tPrec@1 {top1:.3f}'.format(top1=acc)))

if __name__ == '__main__':
    main()