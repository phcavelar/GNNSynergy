import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from layers import GCNEncoder, BilinearDecoder, DSN, Decoder


class pre_DrugGAE(nn.Module):
    """单独cell-line的model"""

    def __init__(self, nfeat, nhid, init, dropout, dhid1, inDrop, Drop):
        super(pre_DrugGAE, self).__init__()
        self.encoder_pos = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_add = GCNEncoder(nfeat, nhid, init, dropout)
        self.encoder_neg = GCNEncoder(nfeat, nhid, init, dropout)
        self.dsn = DSN(in_dim=nhid * 3, n_hidden1=dhid1, n_hidden2=dhid1 * 2, out_dim=dhid1, inDrop=inDrop, drop=Drop)

    def forward(self, x, adj_norm_pos, adj_norm_add, adj_norm_neg):
        # Part1 GCN-based graph encoder
        node_embed_pos = self.encoder_pos(x, adj_norm_pos)
        node_embed_add = self.encoder_add(x, adj_norm_add)
        node_embed_neg = self.encoder_neg(x, adj_norm_neg)

        node_embed = torch.cat([node_embed_pos, node_embed_add], dim=1)
        node_embed = torch.cat([node_embed, node_embed_neg], dim=1)
        node_embed = self.dsn(node_embed)

        return node_embed

class TransferCell(nn.Module):
    def __init__(self, nfeat, nhid, init, dropout, dhid1, similarity, num_cell_lines,
                 cell_line_list, cell_lines_id, now_id, ndim1, ndim2, inDrop, Drop, inDrop_agg, Drop_agg, fold):
        super(TransferCell, self).__init__()
        self.now_id = now_id

        # self.attW = Parameter(torch.empty(len(similarity)))
        # nn.init.normal_(self.attW)
        self.attW = Parameter(torch.FloatTensor(similarity))


        self.num_cell_lines = num_cell_lines


        self.mainview = pre_DrugGAE(nfeat, nhid, init, dropout, dhid1, inDrop, Drop)
        if self.training is True:
            d = cell_lines_id[self.now_id]
            # file = 'Record/LastModel/preTrain_5kfold/' + str(fold) + '_fold/' + '%s_%s' % (d, cell_line_list[d]) + '.pkl'
            file = 'Record/LastModel/preTrain/' + '%s_%s' % (d, cell_line_list[d]) + '.pkl'
            self.mainview.load_state_dict(torch.load(file), strict=False)

        self.subView = nn.ModuleList(
            [pre_DrugGAE(nfeat, nhid, init, dropout, dhid1, inDrop, Drop) for i in range(num_cell_lines - 1)])
        if self.training is True:
            t = 0
            for v in range(num_cell_lines):
                if v != now_id:
                    d = cell_lines_id[v]
                    # file = 'Record/LastModel/preTrain_5kfold/' + str(fold) + '_fold/' + '%s_%s' % (d, cell_line_list[d]) + '.pkl'
                    file = 'Record/LastModel/preTrain/' + '%s_%s' % (d, cell_line_list[d]) + '.pkl'
                    self.subView[t].load_state_dict(torch.load(file), strict=False)
                    for param in self.subView[t].parameters():
                        param.requires_grad = False
                    t += 1

        self.aggregate = DSN(in_dim=dhid1 * (num_cell_lines - 1), n_hidden1=dhid1 * ndim1, n_hidden2=dhid1 * ndim2,
                             out_dim=dhid1, inDrop=inDrop_agg, drop=Drop_agg)
        self.decoder = BilinearDecoder(in_fea=dhid1 * 2, out_fea=dhid1 * 2)
        # self.decoder = BilinearDecoder(in_fea=dhid1 * num_cell_lines, out_fea=dhid1 * num_cell_lines)

    def calSubView(self, x, cell_line_adjs_pos, cell_line_adjs_add, cell_line_adjs_neg):

        self.similarity = F.softmax(self.attW, dim=0)

        isFlag = True
        t = 0
        for v in range(self.num_cell_lines):
            if v != self.now_id and isFlag:
                # subview = torch.mul((1./len(self.similarity)), self.subView[t](x, cell_line_adjs_pos[v], cell_line_adjs_add[v],
                #                                                cell_line_adjs_neg[v]))
                subview = torch.mul(self.similarity[t], self.subView[t](x, cell_line_adjs_pos[v], cell_line_adjs_add[v],
                                                               cell_line_adjs_neg[v]))

                isFlag = False
                t += 1
            elif v != self.now_id:
                # temp = torch.mul((1./len(self.similarity)),self.subView[t](x, cell_line_adjs_pos[v], cell_line_adjs_add[v],
                #                                                cell_line_adjs_neg[v]))
                temp = torch.mul(self.similarity[t], self.subView[t](x, cell_line_adjs_pos[v], cell_line_adjs_add[v],
                                                                       cell_line_adjs_neg[v]))
                subview = torch.cat([subview, temp], dim=1)
                t += 1
        return subview

    def forward(self, x, cell_line_adjs_pos, cell_line_adjs_add, cell_line_adjs_neg):
        # 开始主视角的feature提取
        main_embed = self.mainview(x, cell_line_adjs_pos[self.now_id], cell_line_adjs_add[self.now_id],
                                   cell_line_adjs_neg[self.now_id])

        # 开始辅助视角的feature提取
        subview_embed = self.calSubView(x, cell_line_adjs_pos, cell_line_adjs_add, cell_line_adjs_neg)
        # 将多个子视角feature融合
        subview_embed = self.aggregate(subview_embed)
        # 将主辅视角feature融合
        embed = torch.cat([main_embed, subview_embed], dim=1)
        # 预测
        y_pred = self.decoder(embed)
        return y_pred
