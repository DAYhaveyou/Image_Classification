import os
import logging
import datetime
# from config_load import ActivityConfig as cfg


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            continue
            # self.del_file(c_path)
        else:
            os.remove(c_path)


def _loger(log_dir):
    time_stamp = str(datetime.datetime.now().year)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().month)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().day)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().hour)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().minute)
    time_stamp = time_stamp + "_" + str(datetime.datetime.now().second)

    name = os.path.join(log_dir, time_stamp +".log")
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