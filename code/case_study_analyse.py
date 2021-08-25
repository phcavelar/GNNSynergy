import numpy as np
import pandas as pd
from preprocess import *

# 加载并预处理数据集
# drug_list, drug_fea, cell_line_list, cell_line_fea, tissue_list = getFeature()
# print("Total Data reading ...")
# comb_data_raw = pd.read_csv('../data2/drugComb_process.csv', sep="\t")
#
# # 接下来是找出同个tissue的cell line 序号
# print("获得每个tissue的cell line ids")
# tissue_id_dict = dict()
# for id, key in enumerate(tissue_list):
#     if key in tissue_id_dict.keys():
#         tissue_id_dict[key].append(id)
#     else:
#         tissue_id_dict[key] = [id]
# print(tissue_id_dict)
# # , columns=['drugA', 'drugB', 'cell_line', 'tissue', 'pred_scores']
# # 接下来在特定的tissue里面进行训练
# df = pd.DataFrame({'drugA': 'test',
#                    'drugB': 'test',
#                    'cell_line': 'test',
#                    'tissue': 'test',
#                    'pred_scores': 'test'}, index=[0])
# for tissue, id in tissue_id_dict.items():
#     if tissue == 'soft_tissue':
#         continue
#     print("在{}内训练....".format(tissue))
#     cell_lines_id = id
#     num_cell_lines = len(cell_lines_id)
#
#     for i in range(num_cell_lines):
#         index = i
#         now_id = index
#         # 获取当前cell-line的标号与名字
#         cell_i = cell_lines_id[index]
#         cell_name = cell_line_list[cell_i]
#         print("-" * 40 + '\n')
#         print(" " * 10 + "NO.%s Cell-line \n" % cell_i + " " * 10 + "Cell-line Name:%s" % cell_name)
#         print("-" * 40 + '\n')
#         file = 'Record/LastModel/case_study_predNovel/%s' % cell_i + '.txt'
#         y_pred = np.loadtxt(file)
#         """挑选最大的top10"""
#         for j in range(100):
#             temp = np.where(y_pred == np.max(y_pred))
#             for x in range(len(temp[0])):
#                 row = temp[0][x].item()
#                 col = temp[1][x].item()
#                 if(drug_list[row] == drug_list[col]):
#                     y_pred[row, col] = -5000.
#                     continue
#                 else:
#                     df = df.append({'drugA': drug_list[row],
#                                     'drugB': drug_list[col],
#                                     'cell_line': cell_line_list[cell_i],
#                                     'tissue': tissue,
#                                     'pred_scores': y_pred[row, col]}, ignore_index=True)
#                     y_pred[row, col] = -5000.
#
#         print("选取完成！")
# print("所有细胞系都挑选完成！")
# df.to_csv('Record/LastModel/case_study_predNovel/top_pred.csv')


# 处理每个cell line top预测的药物对
# novel_pred = pd.read_csv('Record/LastModel/case_study_predNovel/top_pred.csv')
# novel_pred = novel_pred.drop(0)
# novel_pred = novel_pred.drop(['Unnamed: 0'], axis=1)
# novel_pred['pred_scores'] = pd.to_numeric(novel_pred['pred_scores'])
# novel_pred.sort_values("pred_scores", inplace=True, ascending=False)
# # novel_pred = novel_pred.drop(novel_pred[novel_pred['drugA']==novel_pred['drugB']].index)
# novel_pred.to_csv('Record/LastModel/case_study_predNovel/top_pred_select.csv')


# 读取Top预测药物组合对的文件
novel_pred = pd.read_csv('Record/LastModel/case_study_predNovel/top_pred_select.csv')
novel_pred = novel_pred.drop(['Unnamed: 0'], axis=1)
# 读取DrugCombDB判定为协同对的文件
# syner_pairs_drugcombdb = pd.read_csv('Record/LastModel/case_study_predNovel/Syner&Antag_voting.csv')
# syner_pairs_drugcombdb = syner_pairs_drugcombdb[syner_pairs_drugcombdb["classification"]=='synergy']
# syner_pairs_drugcombdb = syner_pairs_drugcombdb.reset_index(drop=True)
# index = []
# for i in range(len(novel_pred)):
#     drug1 = novel_pred['drugA'][i]
#     drug2 = novel_pred['drugB'][i]
#     cell = novel_pred['cell_line'][i]
#     for j in range(len(syner_pairs_drugcombdb)):
#         if((drug1==syner_pairs_drugcombdb['Drug1'][j]) and (drug2==syner_pairs_drugcombdb['Drug2'][j]) and (cell==syner_pairs_drugcombdb['Cell line'][j])):
#             index.append(i)
#             print("找到你了！臭崽子！")
#         if ((drug2 == syner_pairs_drugcombdb['Drug1'][j]) and (drug1 == syner_pairs_drugcombdb['Drug2'][j]) and (cell==syner_pairs_drugcombdb['Cell line'][j])):
#             index.append(i)
#             print("找到你了！臭崽子！")
# print(index)
# # 没有排除cell line
# index = [644, 962, 983, 1235, 1968, 2173, 2193, 2286, 2996, 3098, 3098, 3169, 3221, 3243, 3280, 3455, 3502, 3581, 3595, 3630, 3640, 3655, 3655, 3655,
# 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3655, 3658, 3658, 3658, 3658, 3659, 3659, 3659, 3659, 3660, 3660, 3660, 3660,
# 3660, 3660, 3660, 3660, 3660, 3660, 3660, 3660, 3660, 3660, 3660, 3660, 3665, 3665, 3665, 3665, 3665, 3665, 3665, 3665, 3675, 3675, 3675, 3675, 3680, 3680,
# 3680, 3680, 3697, 3697, 3697, 3697, 3697, 3697, 3697, 3697, 3726, 3726, 3726, 3726, 3726, 3726, 3726, 3726, 3726, 3726, 3726, 3740, 3740, 3740, 3740, 3740, 3740,
# 3740, 3740, 3749, 3749, 3749, 3749, 3749, 3749, 3749, 3749, 3749, 3749, 3749, 3756, 3756, 3756, 3756, 3756, 3756, 3756, 3756]
# print("希望有好结果！")
# final = []
# for i in index:
#     if i not in final:
#         final.append(i)
# print(len(index))
# print(len(final))
# df = pd.DataFrame({'drugA': 'test',
#                    'drugB': 'test',
#                 'cell_line': 'test',
#                 'tissue': 'test',
#                 'pred_scores': 'test'}, index=[0])
#
# for i in final:
#     df = df.append({'drugA': novel_pred['drugA'][i],
#                     'drugB': novel_pred['drugB'][i],
#                     'cell_line': novel_pred['cell_line'][i],
#                     'tissue': novel_pred['tissue'][i],
#                     'pred_scores': novel_pred['pred_scores'][i],}, ignore_index=True)
# print("完成了!")
# df.to_csv('Record/LastModel/case_study_predNovel/val_pairs.csv')
import xlrd
# 打开excel
wb = xlrd.open_workbook('Record/LastModel/case_study_predNovel/SynDrugComb_textmining.xlsx')
# 按工作簿定位工作表
sh = wb.sheet_by_name('2drugs')
# print(sh.nrows)#有效数据行数
# print(sh.ncols)#有效数据列数
# print(sh.cell(0,0).value)#输出第一行第一列的值
# print(sh.row_values(0))#输出第一行的所有值
# #将数据和标题组合成字典
# print(dict(zip(sh.row_values(0),sh.row_values(1))))
#遍历excel，打印所有数据
index = []
for j in range(len(novel_pred)):
    drug1 = novel_pred['drugA'][j]
    drug2 = novel_pred['drugB'][j]
    # cell = novel_pred['cell_line'][j]
    for i in range(sh.nrows):
        # print(sh.row_values(i))
        if(drug1 == sh.row_values(i)[0] and drug2 == sh.row_values(i)[1]):
            index.append(j)
            print("找到你了！臭崽子！")
        if (drug2 == sh.row_values(i)[1] and drug1 == sh.row_values(i)[0]):
            index.append(j)
            print("找到你了！臭崽子！")
print(index)



