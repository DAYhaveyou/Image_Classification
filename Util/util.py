import json
import os
import logging
import datetime
import torch
# from config_load import ActivityConfig as cfg


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            continue
        else:
            os.remove(c_path)


def _loger(log_dir, experiment_name):
    '''
    :param log_dir: storage dir of log
    :param experiment_name: atx name of log file
    :return: the object of log
    '''
    time_stamp = str(datetime.datetime.now().year)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().month)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().day)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().hour)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().minute)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().second)
    exp_name = experiment_name+"_"+time_stamp

    name = os.path.join(log_dir, exp_name+".log")
    logger = logging.getLogger("log")
    # logger.disabled = True
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=name,mode='w')
    logger.setLevel(logging.DEBUG)
    # handler1.setLevel(logging.DEBUG)
    # handler2.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s-%(filename)s[line:%(lineno)d]-%(module)s-%(funcName)s-%(levelname)s: %(message)s')
    formatter = logging.Formatter('%(asctime)s-%(filename)s[line:%(lineno)d]: %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def load_config_file(config_file):
    '''
    description: config parameters can be organized in here for further extension.
    :param config_file: dictionary of config file
    :return: the dict of config parameters
    '''
    all_params = json.load(open(config_file))
    # print(all_params)
    return all_params


def accuracy(output, target, topk=(1,)):
    '''
    :param output:  prediction of model
    :param target:  ground-truth labels
    :param topk:  tuple of specified values of k
    :return: an array owns the top k values in tuple topk
    '''
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)      # topk (1,5) maxk = 5
    batch_size = target.size(0)  # batch_size = N

    _, pred = output.topk(maxk, 1, True, True)
    #print(pred.shape) #[N,5]
    pred = pred.t()  # [5,N]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # [5,N]
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CheckpointMeter(object):
    """Storage the best checkpoint's info"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch = 0
        self.acc = 0.0

    def update(self, acc, epoch):
        self.acc = acc
        self.epoch = epoch


def get_pretrain_model(pretrained_file, model):
    '''
    :param pretrained_file: pre-trained model's parameters
    :param model:  model used in training stage
    :return: model that has loaded the pre-trained parameters
    '''
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    pretrained_dict = pre_transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  #
    model.load_state_dict(model_dict)
    return model


def pre_transfer_state_dict(pretrained_dict, model_dict):
    for layer_name, param in pretrained_dict['model_dict'].items():
        # if 'module' in layer_name:
        #     layer_name = layer_name[7:]
        #     if "new_fc" in layer_name:
        #         continue
        if isinstance(param, torch.nn.parameter.Parameter):
            param = param.data
        if layer_name in model_dict.keys():
            model_dict[layer_name].copy_(param)
        else:
            print("Missing key(s) in state_dict :{}".format(layer_name))

    return model_dict