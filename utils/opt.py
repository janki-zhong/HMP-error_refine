#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from utils import log
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:0', help='cuda idx')
        self.parser.add_argument('--data_dir', type=str,
                                 default='E:\code\Python\BARNet\dataset\h3.6m',
                                 help='path to dataset')
        self.parser.add_argument('--rep_pose_dir', type=str,
                                 default='./rep_pose/rep_pose.txt',help='path to dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=1, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=1, help='skip rate of samples for test')
        self.parser.add_argument('--extra_info', type=str, default='', help='extra information') 

        # ===============================================================
        #                     Model options
        # ===============================================================
        # self.parser.add_argument('--input_size', type=int, default=2048, help='the input size of the neural net')
        # self.parser.add_argument('--output_size', type=int, default=85, help='the output size of the neural net')
        self.parser.add_argument('--in_features', type=int, default=66, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=64, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=10, help='past frame number')
        self.parser.add_argument('--kernel_size2', type=int, default=15, help='past frame number')
        self.parser.add_argument('--drop_out', type=float, default=0.3, help='drop out probability')

        self.parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--hidden_size', type=int, default=16,help='Hidden_size//4 = number of quaternion units within each hidden layer.')
        #_---------------------------------------------------
        self.parser.add_argument('--joints_to_consider', type=int, default=22, choices=[16, 18, 22], help='number of joints to use, defaults=16 for H36M angles, 22 for H36M 3D or 18 for AMASS/3DPW')
        self.parser.add_argument('--n_tcnn_layers', type=int, default=4,help='number of layers for the Time-Extrapolator Convolution')
        self.parser.add_argument('--tcnn_kernel_size', type=list, default=[3, 3],help=' kernel for the Time-Extrapolator CNN layers')
        self.parser.add_argument('--tcnn_dropout', type=float, default=0.0, help='tcnn dropout')
        self.parser.add_argument('--aug1', type=str, default='subtract_randomFlip_shear')

        self.parser.add_argument('--aug2', type=str, default='subtract_randomFlip_shear')
        self.parser.add_argument('--num_heads', type=int, default=4, help='size of each model layer')

        self.parser.add_argument('--alpha', type=float, default=0.05, help='the RATIO of the label predict loss')
        # _---------------------------------------------------
        ########mlp_config##############
        # self.parser.add_argument('--hidden_dim', type=int, default=66, help='past frame number')
        # self.parser.add_argument('--seq_len', type=int, default=50, help='past frame number')
        # self.parser.add_argument('--num_layers', type=int, default=48, help='past frame number')
        # self.parser.add_argument('--with_normalization', type=bool, default=True, help='past frame number')
        # self.parser.add_argument('--with_normalization', type=bool, default=False, help='past frame number')

        ########mlp_config##############

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--encoder_n', type=int, default=6, help='encoder layer num')
        self.parser.add_argument('--decoder_n', type=int, default=6, help='decoder layer num')
        self.parser.add_argument('--rep_pose_size', type=int, default=2000, help='rep_pose_size')
        self.parser.add_argument('--updata_rate', type=float, default=0.3, help='rep pose updata_rate')
        self.parser.add_argument('--input_n', type=int, default=10, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=25, help='future frame number')
        self.parser.add_argument('--dct_n', type=int, default=35, help='future frame number')# 使用10帧，预测10帧，时间序列总长20
        self.parser.add_argument('--lr_now', type=float, default=0.005)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=32)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')
        self.parser.add_argument('--test_sample_num', type=int, default=256, help='the num of sample, '
                                                                                  'that sampled from test dataset'
                                                                                  '{8,256,-1(all dataset)}')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, makedir=True):
        self._initial()
        self.opt = self.parser.parse_args()

        # if not self.opt.is_eval:
        script_name = os.path.basename(sys.argv[0])[:-3]
        if self.opt.test_sample_num == -1:
            test_sample_num = 'all'
        else:
            test_sample_num = self.opt.test_sample_num

        if self.opt.test_sample_num == -2:
            test_sample_num = '8_256_all'

        log_name = '{}_{}_in{}_out{}_ks{}_dctn{}_dropout_{}_lr_{}_d_model_{}_e_{}_d_{}'.format(script_name,
                                                                          test_sample_num,
                                                                          self.opt.input_n,
                                                                          self.opt.output_n,
                                                                          self.opt.kernel_size,
                                                                          self.opt.dct_n,
                                                                          self.opt.drop_out,
                                                                          self.opt.lr_now,
                                                                          self.opt.d_model,
                                                                          self.opt.encoder_n,
                                                                          self.opt.decoder_n,
                                                                          )
        self.opt.exp = log_name
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if makedir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)

        self._print()
        # log.save_options(self.opt)
        return self.opt
