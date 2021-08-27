# @Time 2021/3/19 11:56
# @Author JimmyZhan
# @Version:
# @Function:
from __future__ import print_function
from __future__ import division
import argparse
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import torch.optim as optim
from model_dynamic1 import *
from preprocess import *
from evaluate import *
from loss import *

import os
import sys


# sys.path.append('/home/zjm/code/drugComb/gcn_encoder_decoder_cell_all')
# os.chdir(sys.path[-1])


def main():
    # 超参数设置
    # 超参数设置
    parser = argparse.ArgumentParser(description='Model of predict drugComb')

    parser.add_argument('--train_test_mode', type=int, default=1, help='Judeg train or test.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--kfold', type=int, default=1, help='Initial KFold.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--dhid1', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--Drop', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--inDrop', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--Drop_agg', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--inDrop_agg', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--ndim1', type=int, default=5, help='Thresold to divide positive/additive/negative.')
    parser.add_argument('--ndim2', type=int, default=2, help='Thresold to divide positive/additive/negative.')
    parser.add_argument('--thresold', type=int, default=10, help='Thresold to divide positive/additive/negative.')
    parser.add_argument('--top', type=int, default=4, help='Thresold to select top cell-line')
    parser.add_argument('--pcak', type=int, default=10, help='Thresold to select top cell-line')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--file', default='bestWish.pkl', help='Name of save model')
    parser.add_argument('--init', default='Kaiming', help='Name of save model')

    parser.add_argument('--comb_data_name', default='../data2/drugComb_process.csv',
                        help="Name of the drug combination data")

    args = parser.parse_args()

    print("这个程序是fineTue所有tissue的cell line")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    np.random.seed(args.seed)

    # 加载并预处理数据集
    drug_list, drug_fea, cell_line_list, cell_line_fea, tissue_list = getFeature()
    print("Total Data reading ...")
    comb_data_raw = pd.read_csv(args.comb_data_name, sep="\t")

    # 接下来是找出同个tissue的cell line 序号
    print("获得每个tissue的cell line ids")
    tissue_id_dict = dict()
    for id, key in enumerate(tissue_list):
        if key in tissue_id_dict.keys():
            tissue_id_dict[key].append(id)
        else:
            tissue_id_dict[key] = [id]
    print(tissue_id_dict)


    # 接下来在特定的tissue里面进行训练
    for tissue, id in tissue_id_dict.items():
        number_drug_pair = 0
        # if tissue != 'large_intestine':
        #     continue
        print("在{}内训练....".format(tissue))
        print(
            "HyperParams:\nlr:{}\nhidden:{}\ndhid1:{}\ndropout:{}\nDrop:{}\ninDrop:{}\nDrop_agg:{}\ninDrop_agg:{}".format(
                args.lr, args.hidden,
                args.dhid1,
                args.dropout, args.Drop,
                args.inDrop, args.Drop_agg, args.inDrop_agg))
        cell_lines_id = id
        num_cell_lines = len(cell_lines_id)

        # 获得输入的adj矩阵
        cell_line_adjs_neg = []
        cell_line_adjs_add = []
        cell_line_adjs_pos = []

        for i in cell_lines_id:
            with open('input/adj_pos/adj_pos_' + str(i) + '.pickle',
                      'rb') as f:  # file是储存在硬盘中的文件名称
                adj_pos = pickle.load(f)
            with open('input/adj_add/adj_add_' + str(i) + '.pickle',
                      'rb') as f:  # file是储存在硬盘中的文件名称
                adj_add = pickle.load(f)
            with open('input/adj_neg/adj_neg_' + str(i) + '.pickle',
                      'rb') as f:  # file是储存在硬盘中的文件名称
                adj_neg = pickle.load(f)

            adj_pos = adj_pos.cuda()
            adj_add = adj_add.cuda()
            adj_neg = adj_neg.cuda()
            cell_line_adjs_neg.append(adj_neg)
            cell_line_adjs_add.append(adj_add)
            cell_line_adjs_pos.append(adj_pos)

        for i in range(num_cell_lines):
            index = i
            now_id = index
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
                torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子

            # 获取当前cell-line的标号与名字
            cell_i = cell_lines_id[index]
            cell_name = cell_line_list[cell_i]

            # 设置保存模型文件名
            # writer = SummaryWriter('Record/logs/fineTune/tensorboard_' + str(index))
            # args.file = 'Record/logs/fineTune/tensorboard_' + str(index) + '/' + '%s_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
            #     cell_i, cell_name, args.hidden, args.dhid1, args.dropout, args.Drop, args.inDrop, args.lr, args.Drop_agg, args.inDrop_agg) + '.pkl'
            # args.file = 'Record/LastModel/finetune/%s_%s' % (cell_i, cell_name) + '.pkl'

            print("-" * 40 + '\n')
            print(" " * 10 + "NO.%s Cell-line \n" % cell_i + " " * 10 + "Cell-line Name:%s" % cell_name)
            print("-" * 40 + '\n')

            synergies, comb_data, comb_data_pos, comb_data_neg, comb_data_add = data_loader(comb_data_raw,
                                                                                            args.thresold,
                                                                                            cell_name)
            # 统计cell line药物对数量
            number_drug_pair += len(synergies)

            # 获得细胞系之间的相似度
            drug_fea1, cell_line_fea1 = featureNormalize(drug_fea, cell_line_fea)

            # PCA降维
            print("PCA维度为:{}".format(args.pcak))
            pca = PCA(n_components=args.pcak)
            pca.fit(cell_line_fea1)
            cell_line_fea_pca = pca.fit_transform(cell_line_fea1)
            # cell_line_fea_pca = cell_line_fea1

            y_true = synergies

            cell_similartiy = cal_cosine(cell_line_fea_pca)
            # 获得已经排好序的细胞系相似度
            similarity = []
            cell_specific_similiarity = cell_similartiy[cell_i]
            for i in cell_lines_id:
                similarity.append(cell_specific_similiarity[i])
            del similarity[now_id]
            # sort_id = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)

            # similarity = torch.FloatTensor(similarity)
            # if torch.cuda.is_available():
            #     similarity = torch.FloatTensor(similarity).cuda()

            # 获得三个子图的边
            inter_pairs_all, inter_pairs_pos, inter_pairs_neg, inter_pairs_add, num_node = getDrugPairs(drug_list,
                                                                                                        comb_data,
                                                                                                        comb_data_pos,
                                                                                                        comb_data_neg,
                                                                                                        comb_data_add)
            # 分别构建Graph
            # g_all, g_pos, g_neg, g_add = bulidDrugGraph(inter_pairs_all, inter_pairs_pos, inter_pairs_neg,
            #                                             inter_pairs_add,
            #                                             num_node)
            # 获得mask(Train/Test/Val)
            train_ind_kfold, val_ind_kfold, test_ind_kfold = splitData(args.seed, comb_data, args.kfold)

            mse_kfold = []
            spearman_kfold = []
            pearson_kfold = []
            r2_kfold = []
            rmse_kfold = []
            mae_kfold = []
            for j in range(args.kfold):
                # args.file = 'Record/LastModel/finetune_5kfold/' + str(j) + '/%s_%s' % (cell_i, cell_name) + '.pkl'
                # args.file = 'Record/LastModel/attentionModel/attW_fixed/' + '%s_%s' % (cell_i, cell_name) + '.pkl'
                # args.file = 'Record/LastModel/finetune_pca/' + '%s_%s' % (cell_i, cell_name) + '.pkl'
                feature = drug_fea1
                if args.kfold == 1:
                    train_ind = train_ind_kfold
                    val_ind = val_ind_kfold
                    test_ind = test_ind_kfold
                else:
                    train_ind = train_ind_kfold[j]
                    val_ind = val_ind_kfold[j]
                    test_ind = test_ind_kfold[j]
                print("This is the {}th kflod........".format(j))
                train_mask, val_mask, test_mask = bulid_mask(train_ind, val_ind, test_ind, num_node, inter_pairs_all)

                # 获得y_ture矩阵
                y_true_mat = construct_target_matrix(y_true, inter_pairs_all, num_node)
                # Loss函数的权重定义
                _, weight_mat = cal_weight(y_true_mat * train_mask, train_mask, num_node)

                # 模型和优化器定义
                # 加载模型
                model = TransferCell(nfeat=feature.shape[1],
                                     nhid=args.hidden,
                                     init='Kaiming',
                                     dropout=args.dropout,
                                     dhid1=args.dhid1,
                                     similarity=similarity,
                                     num_cell_lines=num_cell_lines,
                                     cell_line_list=cell_line_list,
                                     cell_lines_id=cell_lines_id,
                                     now_id=now_id,
                                     ndim1=args.ndim1,
                                     ndim2=args.ndim2,
                                     inDrop=args.inDrop,
                                     Drop=args.Drop,
                                     inDrop_agg=args.inDrop_agg,
                                     Drop_agg=args.Drop_agg,
                                     fold=j)

                # 打印模型和模型可训练参数
                # print(model)

                if torch.cuda.is_available():  # tensor和变量转到gpu
                    model = model.cuda()
                    # adj_norm_pos = adj_norm_pos.cuda()
                    # adj_norm_add = adj_norm_add.cuda()
                    # adj_norm_neg = adj_norm_neg.cuda()

                    feature = torch.FloatTensor(feature).cuda()
                    train_mask = torch.IntTensor(train_mask).cuda()
                    val_mask = torch.IntTensor(val_mask).cuda()
                    y_true_mat = torch.FloatTensor(y_true_mat).cuda()
                    weight_mat = torch.FloatTensor(weight_mat).cuda()


                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                loss_fn = LossFunction(y_true_mat, weight_mat)
                evaluator = Evaluator(y_true_mat, test_mask)

                # 训练部分
                if (args.train_test_mode == 1):
                    t_total = time.time()

                    for epoch in range(args.epochs):
                        t = time.time()
                        model.train()
                        optimizer.zero_grad()

                        y_pred = model(feature, cell_line_adjs_pos, cell_line_adjs_add, cell_line_adjs_neg)
                        loss_train = loss_fn.cal_loss(y_pred, mask=train_mask)
                        loss_train.backward()
                        optimizer.step()

                        with torch.no_grad():
                            model.eval()
                            y_pred = model(feature, cell_line_adjs_pos, cell_line_adjs_add, cell_line_adjs_neg)
                            loss_val = loss_fn.cal_val_loss(y_pred, val_mask)

                        print('Epoch: {:04d}'.format(epoch + 1),
                              'loss_train: {:.4f}'.format(loss_train.item()),
                              'loss_val: {:.4f}'.format(loss_val.item()),
                              'time: {:.4f}s'.format(time.time() - t))

                        if (epoch == 0):
                            best_val_loss = loss_val.item()
                            torch.save(model.state_dict(), args.file)  # 只保存网络中的参数 (速度快, 占内存少)
                            print("save model")
                            earlystop_count = 0

                        else:
                            if best_val_loss > loss_val.item():
                                best_val_loss = loss_val.item()
                                torch.save(model.state_dict(), args.file)  # 只保存网络中的参数 (速度快, 占内存少)
                                print("save model")
                                earlystop_count = 0

                            if earlystop_count != args.patience:
                                earlystop_count += 1
                            else:
                                print("early stop!!!!")
                                break

                    print("\nOptimization Finished!")
                    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

                # 加载测试的最好模型参数
                model1 = TransferCell(nfeat=drug_fea1.shape[1],
                                      nhid=args.hidden,
                                      init='Kaiming',
                                      dropout=args.dropout,
                                      dhid1=args.dhid1,
                                      similarity=similarity,
                                      num_cell_lines=num_cell_lines,
                                      cell_line_list=cell_line_list,
                                      cell_lines_id=cell_lines_id,
                                      now_id=now_id,
                                      ndim1=args.ndim1,
                                      ndim2=args.ndim2,
                                      inDrop=args.inDrop,
                                      Drop=args.Drop,
                                      inDrop_agg=args.inDrop_agg,
                                      Drop_agg=args.Drop_agg,
                                      fold=j)

                model1.load_state_dict(torch.load(args.file))
                model1 = model1.cuda()
                # 测试部分
                with torch.no_grad():
                    model1.eval()
                    y_pred = model1(feature, cell_line_adjs_pos, cell_line_adjs_add, cell_line_adjs_neg)

                    # confusion_Matrix, acc, auc, aupr, f1, precision, recall = evaluator.eval_classification(y_pred,
                    #                                                                                         pos_thresold=args.thresold)
                    # print("Confusion Matrix:\n{}".format(confusion_Matrix))
                    # print("ACC, AUC, AUPR, F1:{}\t{}\t{}\t{}".format(acc, auc, aupr, f1))
                    # print("precision, recall:{}\t{}".format(precision, recall))

                    mse, spearman, pearson, r2, rmse, mae = evaluator.eval_regression(y_pred)

                    print("mse, spearman, pearson:{}\t{}\t{}".format(mse, spearman, pearson))
                    print("r2, rmse, mae:{}\t{}\t{}".format(r2, rmse, mae))
                    mse_kfold.append(mse)
                    spearman_kfold.append(spearman)
                    pearson_kfold.append(pearson)
                    r2_kfold.append(r2)
                    rmse_kfold.append(rmse)
                    mae_kfold.append(mae)

                # writer.add_hparams({'lr': args.lr, 'Drop_agg': args.Drop_agg, 'inDrop_agg': args.inDrop_agg},
                #                    {'hparam/mse': mse, 'hparam/spearman': spearman, 'hparam/pearson': pearson})

            mse_mean = np.mean(mse_kfold)
            spearman_mean = np.mean(spearman_kfold)
            pearson_mean = np.mean(pearson_kfold)
            r2_mean = np.mean(r2_kfold)
            rmse_mean = np.mean(rmse_kfold)
            mae_mean = np.mean(mae_kfold)
            print("mse_mean, spearman_mean, pearson_mean, r2_mean, rmse_mean:{}\t{}\t{}\t{}\t{}\t{}".format(mse_mean,
                                                                                                            spearman_mean,
                                                                                                            pearson_mean,
                                                                                                            r2_mean,
                                                                                                            rmse_mean,
                                                                                                            mae_mean))

        print("drug pairs数量为："+ str(number_drug_pair))

if __name__ == '__main__':
    main()
