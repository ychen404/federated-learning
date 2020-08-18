#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from __future__ import print_function
from __future__ import division

import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdateRNN
from models.Nets import MLP, CNNMnist, CNNCifar, TrajPreSimple
from models.Fed import FedAvg
from models.test import test_img
from train_simple import run_rnn, RnnParameterData, generate_input_history

import torch.nn as nn
import torch.optim as optim

import os
import json
import time
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')


"""
Example command to run:
python2.7 main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0
"""


"""
Method needed to run the rnn model
"""

def run(args):

    st = time.time()
    w_locals, loss_locals = [], []
    m = max(int(args.frac * args.num_users), 1)
    print("m: {}".format(m))
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    # create local replica
    local = LocalUpdateRNN(args=args)

    # w, loss = local.train(args, data_train, train_idx, data_test, test_idx, 'train', lr, parameters.clip, model, optimizer,
    #                                 criterion, parameters.model_mode, net=copy.deepcopy(net_glob).to(args.device))
    w, loss, avg_acc = local.train(args=args, net=copy.deepcopy(net_glob).to(args.device))
    w_locals.append(copy.deepcopy(w))
    loss_locals.append(copy.deepcopy(loss))
    # update global weights
    w_glob = FedAvg(w_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    # if args.pretrain == 0:
    
    return avg_acc


if __name__ == '__main__':
    # parse args
    
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'foursquare':
        print("Using foursquare dataset")

    else:
        exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'rnn':
        parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)
        argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
        print('*' * 15 + 'start training' + '*' * 15)
        print('model_mode:{} history_mode:{} users:{}'.format(
            parameters.model_mode, parameters.history_mode, parameters.uid_size))
        
        net_glob = TrajPreSimple(parameters=parameters).cuda()
    
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training

    if args.model == 'cnn' or args.model == 'mlp': 
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []

        for iter in range(args.epochs):
            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users: {}".format(idxs_users))
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

        # plot loss curve
        #    plt.figure()
        #    plt.plot(range(len(loss_train)), loss_train)
        #    plt.ylabel('train_loss')
        #    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

        # testing
        # net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # print("Training accuracy: {:.2f}".format(acc_train))
        # print("Testing accuracy: {:.2f}".format(acc_test))
    
    elif args.model == 'rnn':
    #     print(args)
        ours_acc = run(args)


