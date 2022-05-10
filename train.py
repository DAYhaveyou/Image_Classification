# coding=utf8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import time
from DataLoader.data_loader import DogCatImgLoader, train_transform, test_transform
from torch.utils.data import DataLoader
from Util.util import load_config_file, accuracy, AverageMeter, CheckpointMeter, _loger, check_output_path, load_model
from model import Model

# USED GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')


def validation(model, data_loader, logger):
    ''' validate the trained model '''

    top1 = AverageMeter()
    model.eval()
    epoch_steps = len(data_loader)
    iter_loader = iter(data_loader)
    for i in range(epoch_steps):
        _, data, labels = next(iter_loader)
        input = data.cuda()
        label = labels.cuda()
        output = model(input)

        pred = accuracy(output, label)
        top1.update(pred[0].item(), input.size(0))

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f}'.format(top1=top1)))
    return top1.avg


def train(model, train_loader, arg_paras, logger, optimizer, criterion, epoch):
    ''' for training model '''
    # switch to train mode
    top1 = AverageMeter()
    loss_recoder = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    time_stamp = time.time()

    model.train()
    epoch_steps = len(train_loader)
    iter_loader = iter(train_loader)
    for i in range(epoch_steps):
        data_time.update(time.time()-time_stamp)
        # _, data, labels = next(iter_loader)
        value = next(iter_loader)
        data = value['data']
        labels = value['label']
        input = data.cuda()
        label = labels.cuda()

        output = model(input)
        loss = criterion(output, label)

        # update optimizer and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = accuracy(output, label)

        loss_recoder.update(loss.item(), input.size(0))
        top1.update(pred[0].item(), input.size(0))
        batch_time.update(time.time()-time_stamp)

        if i % arg_paras['train_print_freq'] == 0:
            logger.info(('=> Epoch: [{0}][{1}/{2}],\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'All_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        .format(epoch, i, epoch_steps, batch_time=batch_time, data_time=data_time, loss=loss_recoder, top1=top1)))


def save_best_checkpoint(model, dir, arg_paras):
    check_point_name = arg_paras['experiment_name']+"_best.pth.tar"
    save_path = os.path.join(dir, check_point_name)
    torch.save(model.state_dict(), save_path)


def main():
    # load predefined setting
    config_file = 'Config/config.json'
    arg_paras = load_config_file(config_file)

    train_data_loader = DogCatImgLoader(arg_paras['file_path']['train'], arg_paras['annotation_path']["train"],
                                        train_transform(arg_paras['crop_size']))
    val_data_loader = DogCatImgLoader(arg_paras['file_path']['val'], arg_paras['annotation_path']["val"],
                                      test_transform(arg_paras['crop_size'], arg_paras['scale_size']))
    train_loader = DataLoader(dataset=train_data_loader, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_data_loader, batch_size=1, pin_memory=True, shuffle=False)

    #  1) check output dictionary //检查输出文件路径是否存在
    log_dir = check_output_path(arg_paras['output_path'], atx='Logs')
    chk_dir = check_output_path(arg_paras['output_path'], atx='CheckPoints')

    logger = _loger(log_dir, arg_paras['experiment_name'])

    # 2) load model and set optimizer
    model = load_model(arg_paras)
    if arg_paras['GPU']:
        model = model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=arg_paras['learning_rate'], weight_decay=arg_paras['weight_decay'])
    optimizer = optim.SGD(model.parameters(), lr=arg_paras['learning_rate'], weight_decay=arg_paras['weight_decay'])
    if arg_paras['learning_rate_decay']:
        # 在每一个milestone，学习率将乘以0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg_paras['mile_stone'], gamma=0.1)

    # 3） prepare loss function
    criterion = nn.CrossEntropyLoss()

    chk_meter = CheckpointMeter()  # storage the info of best checkpoint
    for e in range(arg_paras['epoch_num']):
        # for train
        train(model, train_loader, arg_paras, logger, optimizer, criterion, e)

        # for val
        if e % arg_paras['val_freq'] == 0:
            prec = validation(model, data_loader=val_loader, logger=logger)
            if chk_meter.acc < prec:
                chk_meter.update(prec, e)
                save_best_checkpoint(model, chk_dir, arg_paras)
        if arg_paras['learning_rate_decay']:
            scheduler.step()

    logger.info(('=> The Best Checkpoint Epoch [{0}]\tPrec@1 {top1.acc:.3f}'.format(chk_meter.epoch, top1=chk_meter)))


if __name__ == '__main__':
    main()

