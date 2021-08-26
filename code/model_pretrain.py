# coding:utf-8
# -*- coding: utf-8 -*-
# @Time 2021/3/19 12:05
# @Author JimmyZhan
# @Version:
# @Function:
from torch.nn.functional import binary_cross_entropy_with_logits

from layers import *


class DrugGAE(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
        super(DrugGAE, self).__init__()
        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_add = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid*3, n_hidden1=dhid1, n_hidden2=dhid1*2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
        self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)

    def forward(self, x, adj_norm_pos, adj_norm_add, adj_norm_neg):
        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj_norm_pos)
        node_embed_add = self.encoder_add(x, adj_norm_add)
        node_embed_neg = self.encoder_neg(x, adj_norm_neg)

        node_embed = torch.cat([node_embed_pos, node_embed_add], dim=1)
        node_embed = torch.cat([node_embed, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)
        y_pred = self.decoder(node_embed)

        return y_pred

# class DrugGAE(nn.Module):
#     """单独cell-line的model"""
#
#     def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
#         super(DrugGAE, self).__init__()
#         self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
#         # self.encoder_add = GCNEncoder(nfeat, nhid, init, dropout)
#         self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
#         self.dsn = DSN(in_dim=nhid*2, n_hidden1=dhid1, n_hidden2=dhid1*2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
#         self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)
#
#     def forward(self, x, adj_norm_pos, adj_norm_add, adj_norm_neg):
#         # Part1 GCN-based graph encoder
#         node_embed_pos = self.encoder_pos(x, adj_norm_pos)
#         # node_embed_add = self.encoder_add(x, adj_norm_add)
#         node_embed_neg = self.encoder_neg(x, adj_norm_neg)
#
#         node_embed = torch.cat([node_embed_pos, node_embed_neg], dim=1)
#         node_embed = self.dsn(node_embed)
#         y_pred = self.decoder(node_embed)
#
#         return y_pred

class DrugGAE_one(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
        super(DrugGAE_one, self).__init__()
        self.encoder = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid, n_hidden1=dhid1, n_hidden2=dhid1*2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
        self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)

    def forward(self, x, adj_norm_pos):
        # Part1 GCN-based graph encoder
        node_embed = self.encoder(x, adj_norm_pos)
        node_embed = self.dsn(node_embed)
        y_pred = self.decoder(node_embed)

        return y_pred

class DrugGAE_two(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
        super(DrugGAE_two, self).__init__()
        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid*2, n_hidden1=dhid1, n_hidden2=dhid1*2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
        self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)

    def forward(self, x, adj_norm_pos, adj_norm_neg):
        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj_norm_pos)
        node_embed_neg = self.encoder_neg(x, adj_norm_neg)

        node_embed = torch.cat([node_embed_pos, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)
        y_pred = self.decoder(node_embed)

        return y_pred