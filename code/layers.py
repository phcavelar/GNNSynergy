#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/19 13:55
# @Author  : JimmyZhan
# @Site    : 
# @File    : layers.py
# @Software: PyCharm

import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, init, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.use_bias = use_bias
        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == 'Xavier':
            fan_in, fan_out = self.weight.shape
            init_range = np.sqrt(6.0 / (fan_in + fan_out))
            self.weight.data.uniform_(-init_range, init_range)

            if self.use_bias:
                torch.nn.init.constant_(self.bias, 0.)

        elif init == 'Kaiming':
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.use_bias:
                fan_in, _ = self.weight.shape
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.use_bias:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        if inputs.is_sparse:
            support = torch.sparse.mm(inputs, self.weight)
        else:
            support = torch.mm(inputs, self.weight)
        outputs = torch.sparse.mm(adj, support)
        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DSN(nn.Module):

    def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim, inDrop, drop):
        super(DSN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.BatchNorm1d(n_hidden1), nn.ReLU(), nn.Dropout(inDrop))
        nn.init.kaiming_normal_(self.fc1[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1[0].bias, 0.1)
        self.fc2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.BatchNorm1d(n_hidden2), nn.ReLU(), nn.Dropout(drop))

        self.fc3 = nn.Linear(n_hidden2, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class SPN(nn.Module):

    def __init__(self, in_dim, n_hidden1, n_hidden2, drop):
        super(SPN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.BatchNorm1d(n_hidden1), nn.ReLU(), nn.Dropout(drop))
        nn.init.kaiming_normal_(self.fc1[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1[0].bias, 0.1)
        # self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.output = nn.Linear(n_hidden1, 2)
        self.drop = drop

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


class GCNEncoder(nn.Module):
    """GCN Encoder model layer for graph embed"""

    def __init__(self, nfeat, nhid, init, dropout):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, init)
        # self.gc2 = GraphConvolution(nhid, 256, init)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)
        # x = self.gc2(x, adj)
        # x = F.relu(x)
        return x


class BilinearDecoder(nn.Module):
    """Bilinear Decoder model layer for link prediction."""

    def __init__(self, in_fea, out_fea, dropout=0.):
        super(BilinearDecoder, self).__init__()
        self.in_fea = in_fea
        self.out_fea = out_fea
        self.dropout = dropout
        self.weight1 = Parameter(torch.FloatTensor(in_fea, out_fea))
        nn.init.xavier_uniform_(self.weight1.data, gain=1.414)

    def forward(self, input):
        # input = F.dropout(input, self.dropout, training=self.training)
        middle = torch.mm(input, self.weight1)
        output = torch.mm(middle, input.t())
        return output

class Decoder(nn.Module):
    def __init__(self, in_fea, out_fea, dropout=0.):
        super(Decoder, self).__init__()
        self.in_fea = in_fea
        self.out_fea = out_fea
        self.dropout = dropout

    def forward(self, input):
        output = torch.mm(input, input.t())
        return output

class pre_DrugGAE(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid1, init, dropout, dhid1, dhid2, dout):
        super(pre_DrugGAE, self).__init__()
        self.encoder_pos = GCNEncoder(nfeat, nhid1, init, dropout)
        self.encoder_add = GCNEncoder(nfeat, nhid1, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid1, init, dropout)

        self.dsn = DSN(in_dim=nhid1 * 3, n_hidden1=dhid1, n_hidden2=dhid2, out_dim=dout, inDrop=0.2, drop=0.5)

    def forward(self, x, adj1, adj2, adj3):
        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj1)
        node_embed_add = self.encoder_add(x, adj2)
        node_embed_neg = self.encoder_neg(x, adj3)

        node_embed = torch.cat([node_embed_pos, node_embed_add], dim=1)
        node_embed = torch.cat([node_embed, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)

        return node_embed
