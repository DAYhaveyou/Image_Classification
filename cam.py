import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from Visual import GradCAM
from Visual import GuidedBackpropReLUModel
from Visual.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image, ImageOps
from model import Model
from Visual.utils.model_targets import ClassifierOutputTarget
from DataLoader.data_loader import ImgCamLoader, train_transform, test_transform, img_transform
from torch.utils.data import DataLoader
from Util.util import load_config_file, accuracy, AverageMeter, CheckpointMeter, _loger, load_model, check_output_path
import os


def img_read(img_dir, img_trans=img_transform()):
    rgb_img = [Image.open(img_dir).convert('RGB')]
    rgb_img = img_trans(rgb_img)
    rgb_img = rgb_img[0]
    rgb_img = np.float32(rgb_img) / 255
    return rgb_img


def show_model_layers_name():
    config_file = 'Config/config.json'
    arg_paras = load_config_file(config_file)
    model = load_model(arg_paras)
    model_dict = model.state_dict()
    for layer_name in model_dict.keys():
        print(layer_name)
    print(model)


def storage_img(cam_image, gb, cam_gb, img_path, atx):
    os.path.join(img_path, f'{atx}_cam.jpg')
    cv2.imwrite(os.path.join(img_path, f'{atx}_cam.jpg'), cam_image)
    cv2.imwrite(os.path.join(img_path, f'{atx}_gb.jpg'), gb)
    cv2.imwrite(os.path.join(img_path, f'{atx}_cam_gb.jpg'), cam_gb)


def split_name(name, s='.'):
    val = str(name).split(s)
    ret_name = val[0]+"_"+val[1]
    return ret_name


def cam_main():
    config_file = 'Config/config.json'
    arg_paras = load_config_file(config_file)

    val_data_loader = ImgCamLoader(arg_paras['file_path']['val'], arg_paras['annotation_path']["negative"],
                                   test_transform(arg_paras['crop_size'], arg_paras['scale_size']))
    val_loader = DataLoader(dataset=val_data_loader, batch_size=1, pin_memory=True, shuffle=False)
    img_root_pth = arg_paras['file_path']['val']

    model = load_model(arg_paras)
    target_layers = [model.base_model.layer4]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=arg_paras['GPU'])

    negative_img_path = os.path.join(arg_paras['output_path'], 'Negative')
    negative_img_path = check_output_path(negative_img_path, arg_paras['experiment_name'])

    model.eval()
    epoch_steps = len(val_loader)
    iter_loader = iter(val_loader)
    for i in range(epoch_steps):
        value = next(iter_loader)
        data = value['data']
        labels = value['label']
        img_dir = value['img_dir']

        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=data,
                            targets=None,
                            eigen_smooth=arg_paras['eigen_smooth'])
        rgb_img = img_read(os.path.join(img_root_pth, img_dir[0]))

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=arg_paras['GPU'])
        gb = gb_model(data, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        storage_img(cam_image, gb, cam_gb, img_path=negative_img_path, atx=split_name(img_dir[0]))
        print('PROCESSED IMG: ', img_dir[0])



if __name__ == '__main__':
    # cam_main()
    # test()
    show_model_layers_name()


