#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/19 13:57
# @Author  : JimmyZhan
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import KFold, train_test_split
import random


def data_loader(comb_data, t, cell_name):
    """
    加载数据集，包括comb-data, chem1, chem2, cell=line data
    :param drug1_chemicals:
    :param drug2_chemicals:
    :param cell_line_gex:
    :param comb_data_name:
    :return:
    """
    # print("Get the specific cell data {}".format(cell_name))
    comb_data = comb_data.drop(['Unnamed: 0'], axis=1)
    comb_data = comb_data[comb_data['cell_line_name'] == cell_name]
    synergies = np.array(comb_data["synergy_loewe"])

    comb_data = comb_data.reset_index(drop=True)

    # 索引都没有改变
    comb_data_pos = comb_data[comb_data["synergy_loewe"] > t]
    comb_data_neg = comb_data[comb_data["synergy_loewe"] < -t]
    comb_data1 = pd.merge(comb_data_pos, comb_data_neg, on=['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe'],
                          how='outer')
    comb_data2 = comb_data.append(comb_data1)
    comb_data_add = comb_data2.drop_duplicates(subset=['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe'],
                                               keep=False)

    return synergies, comb_data, comb_data_pos, comb_data_neg, comb_data_add


def getDrugPairs(drug_list, comb_data, comb_data_pos, comb_data_neg, comb_data_add):
    """预处理得到的数据集得到相互作用的药物对序号"""
    # Map the index for drug
    id_mapping = dict(zip(drug_list, range(len(drug_list))))  # dict:{drug_name: index}
    # Get all the Inter pairs
    inter_pairs = []
    row, col = [], []
    for index in comb_data.index:
        name1, name2 = comb_data.loc[index].values[0].rstrip(), comb_data.loc[index].values[1].rstrip()
        inter_pairs.append((id_mapping[name1], id_mapping[name2]))  # get the drugComb pairs(numbers)
        row.append(id_mapping[name1])
        col.append(id_mapping[name2])
    inter_pairs = np.array(inter_pairs, dtype=np.int32)

    # Get the positive pairs(scores > 10)
    inter_pairs1 = []
    for index in comb_data_pos.index:
        name1, name2 = comb_data_pos.loc[index].values[0].rstrip(), comb_data_pos.loc[index].values[1].rstrip()
        inter_pairs1.append((id_mapping[name1], id_mapping[name2]))  # get the drugComb pairs(numbers)
    inter_pairs1 = np.array(inter_pairs1, dtype=np.int32)

    # Get the negative pairs(scores < -10)
    inter_pairs2 = []
    for index in comb_data_neg.index:
        name1, name2 = comb_data_neg.loc[index].values[0].rstrip(), comb_data_neg.loc[index].values[1].rstrip()
        inter_pairs2.append((id_mapping[name1], id_mapping[name2]))  # get the drugComb pairs(numbers)
    inter_pairs2 = np.array(inter_pairs2, dtype=np.int32)

    # Get the negative pairs(-10 < scores < 10)
    inter_pairs3 = []
    for index in comb_data_add.index:
        name1, name2 = comb_data_add.loc[index].values[0].rstrip(), comb_data_add.loc[index].values[1].rstrip()
        inter_pairs3.append((id_mapping[name1], id_mapping[name2]))  # get the drugComb pairs(numbers)
    inter_pairs3 = np.array(inter_pairs3, dtype=np.int32)
    num_node = len(id_mapping)

    return inter_pairs, inter_pairs1, inter_pairs2, inter_pairs3, num_node


def getFeature():
    """获得drugComb所有的药物list和81个cell-Line以及他们的features"""
    drug_list = []
    with open('../data2/drugList.txt') as inf:
        for line in inf:
            drug_name = line.rstrip()
            drug_list.append(drug_name)
    drug_feature = np.loadtxt('../data2/drugFeatures.txt')

    print("Total have {} drugs!".format(len(drug_list)))
    print("Total have {} feature!".format(len(drug_feature)))

    cell_line_list = []
    tissue_list = []
    with open('../data2/cell_line_list1.txt') as inf:
        for line in inf:
            line = line.rstrip()
            cell_name, tissue = line.split('\t')
            cell_line_list.append(cell_name)
            tissue_list.append(tissue)
    cell_feature = np.loadtxt('../data2/cellFeatures1.txt')

    print("Total have {} cells!".format(len(cell_line_list)))
    print("Total have {} feature!".format(len(cell_feature)))
    return drug_list, drug_feature, cell_line_list, cell_feature, tissue_list


def featureNormalize(drug_fea, cell_fea):
    norm = 'tanh_norm'
    drug_fea, mean1, std1, mean2, std2, feat_filt = normalize(drug_fea, norm=norm)
    cell_fea, mean1, std1, mean2, std2, feat_filt = normalize(cell_fea, norm=norm)
    return drug_fea, cell_fea

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1 != 0
    X = X[:, feat_filt]

    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X - means1) / std1[feat_filt]
    if norm == 'norm':
        return (X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return (np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X - means2) / std2
        X[:, std2 == 0] = 0
        return (X, means1, std1, means2, std2, feat_filt)


def bulidDrugGraph(inter_pairs_all, inter_pairs_pos, inter_pairs_neg, inter_pairs_add, num_node):
    # 构建all图
    g_all = np.zeros((num_node, num_node))
    # Construct a Graph
    for i in range(len(inter_pairs_all)):
        pair_x, pair_y = inter_pairs_all[i, 0], inter_pairs_all[i, 1]
        g_all[pair_x, pair_y] = 1
        g_all[pair_y, pair_x] = 1

    # 构建positive子图
    g_pos = np.zeros((num_node, num_node))
    # Construct a Graph
    for i in range(len(inter_pairs_pos)):
        pair_x, pair_y = inter_pairs_pos[i, 0], inter_pairs_pos[i, 1]
        g_pos[pair_x, pair_y] = 1
        g_pos[pair_y, pair_x] = 1

    # 构建negative子图
    g_neg = np.zeros((num_node, num_node))
    # Construct a Graph
    for i in range(len(inter_pairs_neg)):
        pair_x, pair_y = inter_pairs_neg[i, 0], inter_pairs_neg[i, 1]
        g_neg[pair_x, pair_y] = 1
        g_neg[pair_y, pair_x] = 1

    # 构建additive子图
    g_add = np.zeros((num_node, num_node))
    # Construct a Graph
    for i in range(len(inter_pairs_add)):
        pair_x, pair_y = inter_pairs_add[i, 0], inter_pairs_add[i, 1]
        g_add[pair_x, pair_y] = 1
        g_add[pair_y, pair_x] = 1
    return g_all, g_pos, g_neg, g_add

def bulidDrugGraph_one(inter_pairs_all, num_node):
    # 构建all图
    g_all = np.zeros((num_node, num_node))
    # Construct a Graph
    for i in range(len(inter_pairs_all)):
        pair_x, pair_y = inter_pairs_all[i, 0], inter_pairs_all[i, 1]
        g_all[pair_x, pair_y] = 1
        g_all[pair_y, pair_x] = 1
    return g_all

def normalize_mat(mat, normal_dim):
    # adj = sp.coo_matrix(adj)
    if normal_dim == 'Row&Column':
        # adj_ = mat + sp.eye(mat.shape[0])
        rowsum = np.array(mat.sum(1))
        inv = np.power(rowsum, -0.5).flatten()
        inv[np.isinf(inv)] = 0.
        degree_mat_inv_sqrt = sp.diags(inv)
        # D^{-0.5}AD^{-0.5}
        mat_normalized = mat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return mat_normalized

    elif normal_dim == 'Row':
        rowsum = np.array(mat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mat_normalized = r_mat_inv.dot(mat)
        return mat_normalized


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    :param sparse_mx:
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def splitData(seed, comb_data, kflod):
    # print("Preparation of train/validation/test data.")
    if kflod != 1:
        index_pairs = list(comb_data.index)
        prng = np.random.RandomState(seed)
        kf = KFold(n_splits=kflod, random_state=prng, shuffle=True)

        train_kfold = []
        val_kfold = []
        test_kfold = []
        for train_indices, test_indices in kf.split(index_pairs):
            train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=prng)
            train_kfold.append(train_indices)
            val_kfold.append(val_indices)
            test_kfold.append(test_indices)

        return train_kfold, val_kfold, test_kfold

    else:
        index_pairs = list(comb_data.index)
        prng = np.random.RandomState(seed)
        train_ind, test_ind, val_ind = train_test_val_split(index_pairs, ratio_train=0.6, ratio_test=0.2, ratio_val=0.2,
                                                            random=prng)
        return train_ind, val_ind, test_ind

def splitData_classify(seed, comb_data, kflod):
    # print("Preparation of train/validation/test data.")
    if kflod != 1:
        index_pairs = list(comb_data.index)
        prng = np.random.RandomState(seed)
        kf = KFold(n_splits=kflod, random_state=prng, shuffle=True)

        train_kfold = []
        test_kfold = []
        for train_indices, test_indices in kf.split(index_pairs):
            train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=prng)
            train_kfold.append(train_indices)
            test_kfold.append(test_indices)

        return train_kfold, test_kfold

    else:
        index_pairs = list(comb_data.index)
        prng = np.random.RandomState(seed)
        train_ind, test_ind = train_test_split(index_pairs, test_size=0.2, random_state=prng)
        return train_ind, test_ind

def train_test_val_split(df, ratio_train, ratio_test, ratio_val, random):
    train, middle = train_test_split(df, test_size=1 - ratio_train, random_state=random)
    ratio = ratio_val / (1 - ratio_train)
    test, validation = train_test_split(middle, test_size=ratio, random_state=random)
    return train, test, validation


def bulid_mask(train_ind, val_ind, test_ind, num_node, pairs):
    train_mask = np.zeros((num_node, num_node))
    val_mask = np.zeros((num_node, num_node))
    test_mask = np.zeros((num_node, num_node))
    for i in range(len(train_ind)):
        pair_x_train, pair_y_train = pairs[train_ind[i]][0], pairs[train_ind[i]][1]
        train_mask[pair_x_train, pair_y_train] = 1
        train_mask[pair_y_train, pair_x_train] = 1

    for i in range(len(test_ind)):
        pair_x_test, pair_y_test = pairs[test_ind[i]][0], pairs[test_ind[i]][1]
        test_mask[pair_x_test, pair_y_test] = 1

    for i in range(len(val_ind)):
        pair_x_val, pair_y_val = pairs[val_ind[i]][0], pairs[val_ind[i]][1]
        val_mask[pair_x_val, pair_y_val] = 1

    return train_mask, val_mask, test_mask
    # return train_mask, test_mask
def bulid_mask_classify(train_ind, test_ind, num_node, pairs):
    train_mask = np.zeros((num_node, num_node))
    test_mask = np.zeros((num_node, num_node))
    for i in range(len(train_ind)):
        pair_x_train, pair_y_train = pairs[train_ind[i]][0], pairs[train_ind[i]][1]
        train_mask[pair_x_train, pair_y_train] = 1
        train_mask[pair_y_train, pair_x_train] = 1

    for i in range(len(test_ind)):
        pair_x_test, pair_y_test = pairs[test_ind[i]][0], pairs[test_ind[i]][1]
        test_mask[pair_x_test, pair_y_test] = 1

    return train_mask, test_mask

def construct_target_matrix(score, inter_pairs, num_node):
    """
    构建目标的分数矩阵
    :param score:
    :param inter_pairs:
    :param num_node:
    :return:
    """
    target_matrix = np.zeros((num_node, num_node))
    number = 0
    for i in range(len(inter_pairs)):
        pair_x, pair_y = inter_pairs[i, 0], inter_pairs[i, 1]
        if pair_x != pair_y:
            target_matrix[pair_x, pair_y] = score[i]
            target_matrix[pair_y, pair_x] = score[i]
            number += 1
    # target_matrix = torch.FloatTensor(target_matrix)
    print("Get the {} Scores!!!".format(number))
    return target_matrix

def construct_target_matrix_classifier(score, inter_pairs, num_node):
    """
    构建目标的分数矩阵
    :param score:
    :param inter_pairs:
    :param num_node:
    :return:
    """
    target_matrix = np.zeros((num_node, num_node))
    number = 0
    for i in range(len(inter_pairs)):
        pair_x, pair_y = inter_pairs[i, 0], inter_pairs[i, 1]
        if score[i] >= 10:
            target_matrix[pair_x, pair_y] = 1.
            target_matrix[pair_y, pair_x] = 1.
            number += 1
        else:
            target_matrix[pair_x, pair_y] = 0.
            target_matrix[pair_y, pair_x] = 0.
            number += 1
    # target_matrix = torch.FloatTensor(target_matrix)
    print("Get the {} Scores!!!".format(number))
    return target_matrix

def cal_weight(score, mask, num_node):
    truth_score = get_true_score(score, mask)
    min_s = np.amin(truth_score)
    loss_weight = np.log(truth_score - min_s + np.e)
    loss_weight = torch.from_numpy(loss_weight).float()
    weight = np.zeros((num_node, num_node))
    pairs = mask.nonzero()
    for i in range(len(pairs[0])):
        pairs_x, pairs_y = pairs[0][i], pairs[1][i]
        weight[pairs_x, pairs_y] = loss_weight[i]
        weight[pairs_y, pairs_x] = loss_weight[i]

    return loss_weight, weight


def get_true_score(score, mask):
    pairs = mask.nonzero()
    truth_score = []
    for i in range(len(pairs[0])):
        pairs_x, pairs_y = pairs[0][i], pairs[1][i]
        truth_score.append(score[pairs_x, pairs_y])

    return truth_score


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def cal_cosine(feature):
    from sklearn.metrics.pairwise import cosine_similarity
    """求的是修正后的余弦相似度"""
    fea_mean = feature.mean(axis=1)
    item_mean = feature - fea_mean[:, None]
    cosine = cosine_similarity(item_mean, item_mean)
    return cosine

    # normalized_cosine = (1 + cosine) / 2.0
    # return normalized_cosine

    # cosine = cosine_similarity(feature, feature)
    # return cosine

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED

def resort(cell_line_list, comb_data):
    num = [0] * 81
    for i in range(len(comb_data)):
        for j in range(len(cell_line_list)):
            if comb_data['cell_line_name'][i].rstrip() == cell_line_list[j]:
                num[j] = num[j] + 1
    # 将cell按照数量从多到低进行排序
    sorted_id = sorted(range(len(num)), key=lambda k: num[k], reverse=True)
    return sorted_id
