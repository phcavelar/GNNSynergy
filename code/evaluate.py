#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/19 13:57
# @Author  : JimmyZhan
# @Site    : 
# @File    : evaluate.py
# @Software: PyCharm
from sklearn import metrics
import torch
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, accuracy_score

class Evaluator():
    def __init__(self, y_true_mat, test_mask):
        self.y_true_mat = y_true_mat
        self.test_mask = test_mask
        self.target = self.getScores(self.y_true_mat, self.test_mask)

    def getScores(self, score, mask):
        edges = mask.nonzero()
        y = []
        for i in range(len(edges[0])):
             y.append(score[edges[0][i], edges[1][i]])
        y = np.array(y)
        y = y.flatten()
        y = y.astype('float')
        return y

    def pearson(self, y, pred):
        pear = stats.pearsonr(y, pred)
        pear_value = pear[0]
        pear_p_val = pear[1]
        print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
        return pear_value

    def spearman(self, y, pred):
        spear = stats.spearmanr(y, pred)
        spear_value = spear[0]
        spear_p_val = spear[1]
        print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
        return spear_value

    def mse(self, y, pred):
        err = mean_squared_error(y, pred)
        print("Mean squared error is {}".format(err))
        return err

    def squared_error(self, y, pred):
        errs = []
        for i in range(y.shape[0]):
            err = (y[i] - pred[i]) * (y[i] - pred[i])
            errs.append(err)
        return np.asarray(errs)

    def r2(self, y, pred):
        score = r2_score(y, pred)
        return score

    def rmse(self, y, pred):
        err = np.sqrt(mean_squared_error(y, pred))
        return err
    def mae(self, y, pred):
        err = mean_absolute_error(y, pred)
        return err
    def eval_regression(self, y_pred):
        pred = self.getScores(y_pred, self.test_mask)
        mse = self.mse(self.target, pred)
        spearman = self.spearman(self.target, pred)
        pearson = self.pearson(self.target, pred)
        r2 = self.r2(self.target, pred)
        rmse = self.rmse(self.target, pred)
        mae = self.mae(self.target, pred)
        return mse, spearman, pearson, r2, rmse, mae

    def eval_classification(self, y_pred, pos_thresold):
        # y_pred = torch.sigmoid(y_pred)
        pred = self.getScores(y_pred, self.test_mask)

        target = self.target > pos_thresold
        pred = pred > pos_thresold

        confusion_Matrix = confusion_matrix(target, pred)
        acc = accuracy_score(target, pred)
        f1 = f1_score(target, pred)

        precision, recall, thresholds = precision_recall_curve(target, pred)
        aupr = metrics.auc(recall, precision)
        auc = roc_auc_score(target, pred)
        return confusion_Matrix, acc, auc, aupr, f1, precision, recall

    def eval_acc_auc_aupr(self, y_pred, pos_thresold):
        pred = self.getScores(y_pred, self.test_mask)
        auc_test = roc_auc_score(self.target, pred)
        precision, recall, thresholds = precision_recall_curve(self.target, pred)
        aupr_test = metrics.auc(recall, precision)

        f1_test = f1_score(self.target, pred > pos_thresold)

        return auc_test, aupr_test, f1_test

    def getAllScores(self, y_pred, mask):
        pred_scores = self.getScores(y_pred, mask)
        return pred_scores