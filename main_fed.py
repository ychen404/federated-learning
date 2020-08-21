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
from models.Update import LocalUpdate, LocalUpdateRNN, RnnParameter, RnnData
from models.Nets import MLP, CNNMnist, CNNCifar, TrajPreSimple
from models.Fed import FedAvg
from models.test import test_img
# from train_simple import run_rnn, RnnParameter, generate_input_history, RnnData
from train_simple import run_rnn, generate_input_history

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
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    for user in range(args.num_users):
        # create local replica
        local = LocalUpdateRNN(args=args)
        w, loss, avg_acc = local.train(args=args, net=copy.deepcopy(net_glob).to(args.device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
    # update global weights

    print("Updating global weights")
    w_glob = FedAvg(w_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    # if args.pretrain == 0:

    # print loss
    # loss_avg = sum(loss_locals) / len(loss_locals)
    # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    # loss_train.append(loss_avg)
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
        # parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
        #                           voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
        #                           hidden_size=args.hidden_size, dropout_p=args.dropout_p,
        #                           data_name=args.data_name, lr=args.learning_rate,
        #                           lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
        #                           optim=args.optim, attn_type=args.attn_type,
        #                           clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
        #                           model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)    
        
        rnn_data_1 = RnnData(data_path="/home/local/ASUAD/ychen404/Code/DeepMove_new/data/", data_name="foursquare_nyc_20000_user_1")
        rnn_data_2 = RnnData(data_path="/home/local/ASUAD/ychen404/Code/DeepMove_new/data/", data_name="foursquare_nyc_20000_user_1")
        
        # Loc_size depends on the dataset
        parameters = RnnParameter(loc_size=rnn_data_1.loc_size, loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                data_name=args.data_name, lr=args.learning_rate,
                                lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                optim=args.optim, attn_type=args.attn_type,
                                clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                model_mode=args.model_mode, save_path=args.save_path)
    
        argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
                'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
                'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
                'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
                'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
                'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}

        # Create training and testing data
        
        print('*' * 15 + 'start training' + '*' * 15)        
        print('model_mode:{} history_mode:{}'.format(parameters.model_mode, parameters.history_mode))
        # print('model_mode:{} history_mode:{} users:{}'.format(
        #     parameters.model_mode, parameters.history_mode, parameters.uid_size))
        
        net_glob = TrajPreSimple(parameters=parameters, loc_size=rnn_data_1.loc_size).cuda()
    
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
    
    elif args.model == 'rnn':

        loss_train = []        

        data_train, train_idx = [], []
        data_test, test_idx = [], []
        local = []
         
        for user in range(args.num_users):
            print("user={}".format(user))
            # local[user] = LocalUpdateRNN(args=args)
            local.append(LocalUpdateRNN(args=args))
            
            data_train_tmp, train_idx_tmp = generate_input_history(rnn_data_1.data_neural, 'train', mode2=parameters.history_mode,
                                                        candidate=rnn_data_1.data_neural.keys())
            data_train.append(data_train_tmp)
            train_idx.append(train_idx_tmp)
            data_test_tmp, test_idx_tmp = generate_input_history(rnn_data_1.data_neural, 'test', mode2=parameters.history_mode,
                                                        candidate=rnn_data_1.data_neural.keys())
            data_test.append(data_test_tmp)
            test_idx.append(test_idx_tmp)

        print(local)
        for iter in range(args.epochs):
            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
            for user in range(args.num_users):
                # create local replica
                print(30*'*')
                print("User = {}".format(user))
                print(30*'*')
                # local = LocalUpdateRNN(args=args)
                w, loss, avg_acc = local[user].train(args, copy.deepcopy(net_glob).to(args.device), parameters, data_train[user], train_idx[user], data_test[user], test_idx[user])
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                # update global weights
            print("Updating global weights")
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # if args.pretrain == 0:
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print(30*'*')
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)