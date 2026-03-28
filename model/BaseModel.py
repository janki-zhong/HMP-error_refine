#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import sys
import numpy as np
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    图卷积层，结合了空间图卷积和时间维度全连接
    输入形状: [batch_size, in_channels, num_nodes, seq_len]
    输出形状: [batch_size, out_channels, num_nodes, seq_len]
    """
    def __init__(self, in_c, out_c, node_n=22, node_n_s2=10, node_n_s3=5, seq_len=35, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        
        # 可学习的邻接矩阵参数
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        
        # 时间维度的全连接权重
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))
        
        # 通道变换权重
        self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))
        
        # 偏置项
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        stdv = 1. / math.sqrt(self.att.size(1))
        self.weight_c.data.uniform_(-stdv, stdv)
        self.weight_seq.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        前向传播
        1. 根据输入节点数选择对应的邻接矩阵
        2. 执行图卷积操作
        3. 执行时间维度的全连接操作
        """
        support = torch.matmul(self.att, input.permute(0, 3, 2, 1))
        output_gcn = torch.matmul(support, self.weight_c)
        output_fc = torch.matmul(output_gcn.permute(0, 2, 3, 1), self.weight_seq).permute(0, 2, 1, 3).contiguous()
        return output_fc + self.bias if self.bias is not None else output_fc

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'
    
class GCN_decoder(nn.Module):
    def __init__(self, in_channal, out_channal,
                 n_txcnn_layers,txc_kernel_size,txc_dropout,
                 node_n=22,  node_n_s2=10, node_n_s3=5,seq_len=20, p_dropout=0.3, num_stage=1,snum_stage=2,):
        """
        GCN解码器
        输入形状: [batch_size, in_channels, num_nodes, seq_len] == [bs, in_channal, self.node_n, dct_n]
        输出形状: [batch_size, 3, num_nodes, seq_len]
        """
        super(GCN_decoder, self).__init__()
        self.num_stage = num_stage
        self.input_time_frame = seq_len
        self.output_time_frame = seq_len
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()
        self.conv16_3 = torch.nn.Conv2d(in_channels=in_channal, out_channels=3, kernel_size=(1, 1))
        # ----------------------------------
        # self.q4gnn1 = Q4GNNLayer(in_features=in_channal, out_features=in_channal,  node_n=node_n,seq_len=seq_len, dropout=p_dropout)
        # ----------------------------------

        self.txcnns.append(CNN_layer(seq_len, seq_len, txc_kernel_size,
                                     txc_dropout))  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(CNN_layer(seq_len, seq_len, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        # x : [bs, in_channels, self.node_n, dct_n]
        x = x.permute(0, 3, 1, 2)  # [bs, dct_n, in_channels, self.node_n]

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # 残差连接
        x = x.permute(0, 2, 3, 1) # [bs, in_channels, self.node_n, dct_n]
        x = self.conv16_3(x) # [bs, in_channels, self.node_n, dct_n]
        return x

class CNN_layer(nn.Module):
    """
    简单的CNN层，保持输入维度不变（特征维度除外）
    用于时间维度的处理
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout, bias=True):
        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x):
        return self.block(x)