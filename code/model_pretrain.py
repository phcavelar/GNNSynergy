# coding:utf-8
# -*- coding: utf-8 -*-
# @Time 2021/3/19 12:05
# @Author JimmyZhan
# @Version:
# @Function:
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from layers import *


class DrugGAE(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
        super(DrugGAE, self).__init__()
        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_add = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid * 3, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
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


class DrugGAE_one(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
        super(DrugGAE_one, self).__init__()
        self.encoder = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
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
        self.dsn = DSN(in_dim=nhid * 2, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
        self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)

    def forward(self, x, adj_norm_pos, adj_norm_neg):
        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj_norm_pos)
        node_embed_neg = self.encoder_neg(x, adj_norm_neg)

        node_embed = torch.cat([node_embed_pos, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)
        y_pred = self.decoder(node_embed)

        return y_pred

class DrugGAE_two_smile(nn.Module):
    """单独cell-line的model + drug smile graph features"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop, nfeat_smile, nhid_smile, nhid1_smile, smileNode):
        super(DrugGAE_two_smile, self).__init__()
        self.drugSmileGCNEncoder = nn.ModuleList(
            [MolecularGCNEncoder(nfeat_smile, nhid_smile, nhid1_smile, init, dropout, smileNode[i]) for i in range(3040)]
        )
        # """define all embeding here"""
        self.token_size = 43
        self.embedding_tokens = nn.Parameter(torch.empty((self.token_size, nfeat_smile)))
        nn.init.normal_(self.embedding_tokens)

        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid * 2, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
        self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)

    def calDrugSmileFeatures(self, mol_graphs, encoded_drug):
        emb_drugs = []
        for j in range(3040):
            adj = mol_graphs[j]
            drug_sequence = encoded_drug[j]
            inputs = self.get_features(drug_sequence)

            drugSmileFeature =  self.drugSmileGCNEncoder[j](inputs, adj)
            emb_drugs.append(torch.flatten(drugSmileFeature).reshape(1,-1))
        emb_drugs = torch.cat(emb_drugs, 0)
        return emb_drugs

    def get_features(self, sequence):
        sequence = sequence.reshape([1, len(sequence)])
        sequence = sequence.long()
        # embedding = torch.index_select(self.embedding_tokens, 0, sequence)
        embedding = self.embedding_lookup(self.embedding_tokens, sequence)
        return embedding

    def embedding_lookup(self, embedding_token, sequence):
        for index in range(len(sequence[0])):
            if index == 0:
                embedding = embedding_token[sequence[0][index]].reshape([1,-1])
            else:
                embedding = torch.cat([embedding,embedding_token[sequence[0][index]].reshape([1,-1])], 0)

        return embedding

    def forward(self, x, adj_norm_pos, adj_norm_neg, mol_graphs, encoded_drug):
        # Obtain the drug Smile Structure
        emb_drugs = self.calDrugSmileFeatures(mol_graphs, encoded_drug)
        # emb_drugs = torch.nn.functional.normalize(emb_drugs, p=2, dim=1)

        # concat the smile and chemical
        x = torch.cat((x, emb_drugs), 1)
        # x = emb_drugs

        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj_norm_pos)
        node_embed_neg = self.encoder_neg(x, adj_norm_neg)
        node_embed = torch.cat([node_embed_pos, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)
        y_pred = self.decoder(node_embed)

        return y_pred

class MolecularGCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nhid1, init, dropout, NodeNumbers):
        super(MolecularGCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, init)
        self.gc2 = GraphConvolution(nhid, nhid1, init)
        self.dropout = dropout
        self.maxPool = nn.MaxPool2d(kernel_size=(NodeNumbers,1), stride=1)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        # x = F.relu(x)
        x = x.unsqueeze(0)
        x = self.maxPool(x)
        return x


class DrugGAE_classify(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop):
        super(DrugGAE_classify, self).__init__()
        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_add = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid * 3, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
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
        y_pred = torch.sigmoid(y_pred)
        return y_pred

class DrugGAE_only_smile(nn.Module):
    """单独cell-line的model + drug smile graph features"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, Drop, inDrop, nfeat_smile, nhid_smile, nhid1_smile, smileNode):
        super(DrugGAE_only_smile, self).__init__()
        self.drugSmileGCNEncoder = nn.ModuleList(
            [MolecularGCNEncoder(nfeat_smile, nhid_smile, nhid1_smile, init, dropout, smileNode[i]) for i in range(3040)]
        )
        # """define all embeding here"""
        self.token_size = 43
        self.embedding_tokens = nn.Parameter(torch.empty((self.token_size, nfeat_smile)))
        nn.init.normal_(self.embedding_tokens)

        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid * 2, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)
        self.decoder = BilinearDecoder(in_fea=dhid1, out_fea=dhid1)

    def calDrugSmileFeatures(self, mol_graphs, encoded_drug):
        emb_drugs = []
        for j in range(3040):
            adj = mol_graphs[j]
            drug_sequence = encoded_drug[j]
            inputs = self.get_features(drug_sequence)

            drugSmileFeature =  self.drugSmileGCNEncoder[j](inputs, adj)
            emb_drugs.append(torch.flatten(drugSmileFeature).reshape(1,-1))
        emb_drugs = torch.cat(emb_drugs, 0)
        return emb_drugs

    def get_features(self, sequence):
        sequence = sequence.reshape([1, len(sequence)])
        sequence = sequence.long()
        # embedding = torch.index_select(self.embedding_tokens, 0, sequence)
        embedding = self.embedding_lookup(self.embedding_tokens, sequence)
        return embedding

    def embedding_lookup(self, embedding_token, sequence):
        for index in range(len(sequence[0])):
            if index == 0:
                embedding = embedding_token[sequence[0][index]].reshape([1,-1])
            else:
                embedding = torch.cat([embedding,embedding_token[sequence[0][index]].reshape([1,-1])], 0)

        return embedding

    def forward(self, x, adj_norm_pos, adj_norm_neg, mol_graphs, encoded_drug):
        # Obtain the drug Smile Structure
        emb_drugs = self.calDrugSmileFeatures(mol_graphs, encoded_drug)
        # emb_drugs = torch.nn.functional.normalize(emb_drugs, p=2, dim=1)

        # concat the smile and chemical
        # x = torch.cat((x, emb_drugs), 1)
        x = emb_drugs

        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj_norm_pos)
        node_embed_neg = self.encoder_neg(x, adj_norm_neg)
        node_embed = torch.cat([node_embed_pos, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)
        y_pred = self.decoder(node_embed)

        return y_pred