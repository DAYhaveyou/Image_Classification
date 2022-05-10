# coding=utf8
import torch
import numpy as np
import os
import time
from DataLoader.data_loader import DogCatImgLoader, train_transform, test_transform
from torch.utils.data import DataLoader
from Util.feature_extractor import *
from Util.util import load_config_file, accuracy, AverageMeter, CheckpointMeter, _loger, check_output_path, load_model

from sklearn.manifold import TSNE as TSNE1
from tsnecuda import TSNE as TSNE_GPU
import matplotlib.pyplot as plt

# USED GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')


def extract_fea_main():
    # load predefined setting
    config_file = 'Config/config.json'
    arg_paras = load_config_file(config_file)
    val_data_loader = DogCatImgLoader(arg_paras['file_path']['val'], arg_paras['annotation_path']["val"],
                                      test_transform(arg_paras['crop_size'], arg_paras['scale_size']))
    val_loader = DataLoader(dataset=val_data_loader, batch_size=1, pin_memory=True, shuffle=False)
    extract_feature(arg_paras, val_loader)


def show_img(data_dict, path, img_name):
    plt.figure(figsize=(8, 8))
    for key in data_dict.keys():
        data = np.array(data_dict[key])
        plt.scatter(data[:, 0], data[:, 1], c=plt.cm.Set1(int(key)), label="class: "+key)
        # plt.scatter(data[:, 0], data[:, 1], label="class: " + key)
    # plt.show()
    plt.legend()
    if path is None:
        plt.savefig(img_name+'.jpg')
    else:
        plt.savefig(os.path.join(path, img_name + '.jpg'))


def test_plot():
    data_dict = np.load('tempfile.npy')
    show_img(data_dict, path=None, img_name='test')


def tsne_main():
    config_file = 'Config/config.json'
    arg_paras = load_config_file(config_file)
    feature_path = check_output_path(arg_paras['output_path'], atx='Features')
    data, labels = load_feature2_npy(arg_paras, feature_path)

    if arg_paras['tsne_gpu']:
        tsne = TSNE_GPU(n_components=2, learning_rate=10, n_iter=300)
    else:
        tsne = TSNE1(n_components=2, perplexity=15, learning_rate=10, init='pca', n_iter=300, random_state=1024)
    print("Start Fit")
    start_time = time.time()
    x_emb = tsne.fit_transform(data)
    print('Fitting time is: {t:.3f}'.format(t=time.time()-start_time))

    data_dict = {}
    for i in range(len(labels)):
        if str(labels[i]) not in data_dict.keys():
            data_dict[str(labels[i])] = []
        data_dict[str(labels[i])].append(x_emb[i])

    # show_img(x_emb, labels)
    # np.save('tempfile', data_dict)
    visual_path = os.path.join(arg_paras['output_path'], 'Visuals')
    visual_path = check_output_path(visual_path, atx='TSNE')
    show_img(data_dict, visual_path, arg_paras['experiment_name']+"_tsne")


if __name__ == '__main__':
    # main()
    tsne_main()
    pass

