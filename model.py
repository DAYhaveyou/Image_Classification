import os
import sys
import torch

from Model_Zoom import *


class Model():
    """
    select the model that defined in model zoom
    """
    def __init__(self, arg):
        self.arg = arg
        pass

    def select_model(self, name):
        if name == "resnet":
            return resnet18(num_classes=self.arg['num_class'])

        elif name == "senet":
            return se_resnet18(num_classes=self.arg['num_class'])

        elif name == "basenet":
            return BaseNet(num_class=self.arg['num_class'], dp=self.arg['dropout'], arg_paras=self.arg)

        elif name == "senet":
            return SENet(num_class=self.arg['num_class'], dp=self.arg['dropout'])

        else:
            print('No related model!')
            exit(0)