#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 22:06
# @Author  : JimmyZhan
# @Site    : 
# @File    : loss.py
# @Software: PyCharm
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits


class LossFunction():
    def __init__(self, y_true_mat, weight):
        self.y_true = y_true_mat
        self.weight = weight

    def cal_loss(self, y_pred, mask):
        mask = mask.float()
        y_true_mask = torch.mul(self.y_true, mask)
        y_pred_mask = torch.mul(y_pred, mask)

        num_pairs = len(torch.nonzero(mask, as_tuple=False))
        ret = (y_true_mask - y_pred_mask) ** 2
        # loss = torch.sum(torch.mul(self.weight, ret)) / (2*num_pairs)
        loss = torch.sum(ret) / (2*num_pairs)
        return loss

    def cal_val_loss(self, y_pred, mask):
        mask = mask.float()
        y_true_mask = torch.mul(self.y_true, mask)
        y_pred_mask = torch.mul(y_pred, mask)

        num_pairs = len(torch.nonzero(mask, as_tuple=False))
        ret = (y_true_mask - y_pred_mask) ** 2
        loss = torch.sum(ret) / num_pairs
        return loss

    def calStructLoss(self, node_embed_pos, node_embed_add, node_embed_neg,g_pos, g_add, g_neg):
        effect_adj_pos = torch.mm(node_embed_pos, node_embed_pos.t())
        effect_adj_add = torch.mm(node_embed_add, node_embed_add.t())
        effect_adj_neg = torch.mm(node_embed_neg, node_embed_neg.t())

        num_nodes = g_pos.shape[0]
        # num_edges = (len(effect_adj.nonzero()) - num_nodes) / 2 + num_nodes
        num_edges = g_pos.sum()
        effect_weight = float(num_nodes ** 2 - num_edges) / num_edges

        norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)
        total_loss = norm * binary_cross_entropy_with_logits(effect_adj_pos, g_pos,
                                                              pos_weight=effect_weight,
                                                              reduction='mean')

        num_nodes = g_add.shape[0]
        # num_edges = (len(effect_adj.nonzero()) - num_nodes) / 2 + num_nodes
        num_edges = g_add.sum()
        effect_weight = float(num_nodes ** 2 - num_edges) / num_edges

        norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)
        total_loss += norm * binary_cross_entropy_with_logits(effect_adj_add, g_add,
                                                             pos_weight=effect_weight,
                                                             reduction='mean')
        num_nodes = g_neg.shape[0]
        # num_edges = (len(effect_adj.nonzero()) - num_nodes) / 2 + num_nodes
        num_edges = g_neg.sum()
        effect_weight = float(num_nodes ** 2 - num_edges) / num_edges

        norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)
        total_loss = norm * binary_cross_entropy_with_logits(effect_adj_neg, g_neg,
                                                             pos_weight=effect_weight,
                                                             reduction='mean')
        return total_loss

class LossFunction_Classify():
    def __init__(self, target_adj):
        num_edges = target_adj.sum()
        num_nodes = target_adj.shape[0]
        self.pos_weight = float(num_nodes ** 2 - num_edges) / num_edges
        self.norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)
        self.target = target_adj

    def cal_loss(self, logit1, mask):
        # y_true_mask = torch.mul(self.target, mask)
        # y_pred_mask = torch.mul(logit1, mask)

        loss = self.norm * binary_cross_entropy_with_logits(logit1, self.target, pos_weight=self.pos_weight, reduction='mean')

        return loss

# def calTrainLoss(names,cell_lines_id,now_id):
#     celllines_loss = 0
#     for cell_index in range(len(cell_lines_id)):
#         if cell_index != now_id:
#             d = cell_lines_id[cell_index]
#             with open('input/y_trues/y_true_' + str(d) + '.pickle', 'rb') as f:  # file是储存在硬盘中的文件名称
#                 y_true = pickle.load(f)
#             with open('input/train_masks/train_mask_' + str(d) + '.pickle', 'rb') as f:  # file是储存在硬盘中的文件名称
#                 mask = pickle.load(f)
#             with open('input/weight_mats/weight_mat_' + str(d) + '.pickle', 'rb') as f:  # file是储存在硬盘中的文件名称
#                 weight = pickle.load(f)
#
#             y_true = torch.FloatTensor(y_true).cuda()
#             mask = torch.FloatTensor(mask).cuda()
#             weight = torch.FloatTensor(weight).cuda()
#
#             y_pred = names['sub_pred_'+str(cell_index)]
#             y_true_mask = torch.mul(y_true, mask)
#             y_pred_mask = torch.mul(y_pred, mask)
#
#             num_pairs = len(torch.nonzero(mask, as_tuple=False))
#             ret = (y_true_mask - y_pred_mask) ** 2
#             loss = torch.sum(torch.mul(weight, ret)) / num_pairs
#             celllines_loss += loss
#     return celllines_loss/len(cell_lines_id)
#
# def calValLoss(celllines_predict,cell_lines_id):
#
#     celllines_loss = 0
#     for cell_index, y_pred in enumerate(celllines_predict):
#         cell_index = cell_lines_id[cell_index]
#         with open('input/y_trues/y_true_' + str(cell_index) + '.pickle', 'rb') as f:  # file是储存在硬盘中的文件名称
#             y_true = pickle.load(f)
#         with open('input/val_masks/val_mask_' + str(cell_index) + '.pickle', 'rb') as f:  # file是储存在硬盘中的文件名称
#             mask = pickle.load(f)
#
#         y_true_mask = y_true * mask
#         y_pred_mask = y_pred * mask
#
#         num_pairs = len(mask.nonzero()[0])
#         ret = (y_true_mask - y_pred_mask) ** 2
#         loss = np.sum(ret) / num_pairs
#         celllines_loss += loss
#     return celllines_loss / len(celllines_predict)

