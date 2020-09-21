#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
# from sklearn import metrics
from torch.autograd import Variable
from collections import deque, Counter
import torch.optim as optim


import os
import json
import time
import argparse
import numpy as np
from json import encoder
# import cPickle as pickle
import pickle

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        # Sets the module in training mode.        
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class RnnData(object):
    def __init__(self, data_path='../data/', data_name='pub_700'):
        self.data_path = data_path
        self.data_name = data_name

        data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'))

        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']
        
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)

class RnnParameter(object):
    def __init__(self, loc_size, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='../data/', save_path='../results/', data_name='foursquare', accuracy_mode='top1'):

        # self.data_path = data_path
        self.save_path = save_path
        # self.data_name = data_name
        # print(data_name)

        # data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'))
        # data = pickle.load(open(self.data_path + self.data_name + '.pkl', 'rb'))
        # print("The pickled data is {}".format(data))
        # self.vid_list = data['vid_list']
        # self.uid_list = data['uid_list']
        # self.data_neural = data['data_neural']
        self.tim_size = 48
        # self.loc_size = len(self.vid_list)
        # self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode
        self.accuracy_mode = accuracy_mode

class LocalUpdateRNN(object):
    def __init__(self, args):
        self.args = args
        self.selected_clients = []

    def generate_queue(self, train_idx, mode, mode2):
        """return a deque. You must use it by train_queue.popleft()"""
        user = list(train_idx.keys())
        train_queue = deque()
        if mode == 'random':
            initial_queue = {}
            for u in user:
                if mode2 == 'train':
                    initial_queue[u] = deque(train_idx[u][1:])
                else:
                    initial_queue[u] = deque(train_idx[u])
            queue_left = 1
            while queue_left > 0:
                np.random.shuffle(user)
                for j, u in enumerate(user):
                    if len(initial_queue[u]) > 0:
                        train_queue.append((u, initial_queue[u].popleft()))
                    if j >= int(0.01 * len(user)):
                        break
                queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
        elif mode == 'normal':
            for u in user:
                for i in train_idx[u]:
                    train_queue.append((u, i))
        return train_queue

    def get_acc(self, target, scores):
        """target and scores are torch cuda Variable"""
        target = target.data.cpu().numpy()
        
        val, idxx = scores.data.topk(10, 1)
        predx = idxx.cpu().numpy()
    
        acc = np.zeros((3, 1))
        for i, p in enumerate(predx):
            # pdb.set_trace()
            t = target[i]
            if t in p[:10] and t > 0:
                acc[0] += 1
            if t in p[:5] and t > 0:
                acc[1] += 1
            if t == p[0] and t > 0:
                acc[2] += 1
        return acc

    def run_rnn(self, net, data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None, accuracy_mode='top1'):
        """mode=train: return model, avg_loss
        mode=test: return avg_loss,avg_acc,users_rnn_acc"""
        
        run_queue = None
        if mode == 'train':
            model.train(True)
            run_queue = self.generate_queue(run_idx, 'random', 'train')
        elif mode == 'test':
            model.train(False)
            run_queue = self.generate_queue(run_idx, 'normal', 'test')
        total_loss = []
        queue_len = len(run_queue)

        users_acc = {}
        for c in range(queue_len):
            optimizer.zero_grad()
            u, i = run_queue.popleft()
            if u not in users_acc:
                users_acc[u] = [0, 0]
            loc = data[u][i]['loc'].cuda()
            tim = data[u][i]['tim'].cuda()
            target = data[u][i]['target'].cuda()
            uid = Variable(torch.LongTensor([u])).cuda()

            # if mode2 in ['simple', 'simple_long']:
            scores = model(loc, tim)
            
            if scores.data.size()[0] > target.data.size()[0]:
                scores = scores[-target.data.size()[0]:]
            loss = criterion(scores, target)

            if mode == 'train':
                loss.backward()
                # gradient clipping
                try:
                    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                    for p in model.parameters():
                        if p.requires_grad:
                            p.data.add_(-lr, p.grad.data)
                except:
                    pass
                optimizer.step()
            elif mode == 'test':
                users_acc[u][0] += len(target)
                acc = self.get_acc(target, scores)

                if accuracy_mode == 'top5':
                    users_acc[u][1] += acc[1]   
                elif accuracy_mode == 'top1':
                    users_acc[u][1] += acc[2]

            # fixed indices problem
            total_loss.append(loss.data.cpu().numpy())

        avg_loss = np.mean(total_loss, dtype=np.float64)
        if mode == 'train':
            return net, avg_loss
        elif mode == 'test':
            users_rnn_acc = {}
            for u in users_acc:
                tmp_acc = users_acc[u][1] / users_acc[u][0]

                users_rnn_acc[u] = tmp_acc.tolist()[0]
                # print("users_rnn_acc[u]: {}".format(users_rnn_acc[u]))
            avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
            return avg_loss, avg_acc, users_rnn_acc

    def train(self, args, net, parameters, training_data, training_idx, testing_data, testing_idx):
        
        # Sets the module in training mode.        
        SAVE_PATH = args.save_path
        tmp_path = 'checkpoint/'

        argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
        print('*' * 15 + 'start training' + '*' * 15)
        print('model_mode:{} history_mode:{}'.format(parameters.model_mode, parameters.history_mode))
        net.train()
        
        metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}
        
        lr = parameters.lr
        
        # train and update
        criterion = nn.NLLLoss().cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                            factor=parameters.lr_decay, threshold=1e-3)

        for epoch in range(parameters.epoch):
            st = time.time()
            if args.pretrain == 0:
                # _, avg_loss = self.run_rnn(net, data_train, train_idx, 'train', lr, parameters.clip, net, optimizer, criterion, parameters.model_mode)
                _, avg_loss = self.run_rnn(net, training_data, training_idx, 'train', lr, parameters.clip, net, optimizer, criterion, parameters.model_mode)
                print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
                metrics['train_loss'].append(avg_loss)

            # avg_loss, avg_acc, users_acc = self.run_rnn(net, data_test, test_idx, 'test', lr, parameters.clip, net,
            #                                      optimizer, criterion, parameters.model_mode)
            avg_loss, avg_acc, users_acc = self.run_rnn(net, testing_data, testing_idx, 'test', lr, parameters.clip, net,
                                                 optimizer, criterion, parameters.model_mode)
            print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))
        
            metrics['valid_loss'].append(avg_loss)
            metrics['accuracy'].append(avg_acc)
            metrics['valid_acc'][epoch] = users_acc


            save_name_tmp = 'ep_' + str(epoch) + '.m'
            if not os.path.exists(SAVE_PATH + tmp_path):
                os.mkdir(SAVE_PATH + tmp_path)

            torch.save(net.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)
            scheduler.step(avg_acc)
            lr_last = lr
            lr = optimizer.param_groups[0]['lr']
            if lr_last > lr:
                load_epoch = np.argmax(metrics['accuracy'])
                load_name_tmp = 'ep_' + str(load_epoch) + '.m'
                net.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
                print('load epoch={} model state'.format(load_epoch))
            if epoch == 0:
                print('single epoch time cost:{}'.format(time.time() - st))
            if lr <= 0.9 * 1e-5:
                break
            if args.pretrain == 1:
                break
            
        mid = np.argmax(metrics['accuracy'])
        # print(metrics['accuracy'])
        # print("mid: {}".format(mid))
        
        avg_acc = metrics['accuracy'][mid]
        load_name_tmp = 'ep_' + str(mid) + '.m'
        net.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
        save_name = 'res'
        json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
        metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    
        return net.state_dict(), avg_loss, avg_acc

