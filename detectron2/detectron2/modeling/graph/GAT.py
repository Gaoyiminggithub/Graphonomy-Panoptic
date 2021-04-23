import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.activation = activation

        self.W = nn.Linear(in_features, out_features, bias=False)
        weight_init.c2_xavier_fill(self.W)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.adj_func = nn.Linear(2*out_features, 1, bias=False)
        weight_init.c2_xavier_fill(self.adj_func)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj=None):
        hidden = self.W(input)
        # h = torch.mm(input, self.W)
        N = hidden.size()[0]

        a_input = torch.cat([hidden.repeat(1, N).view(N * N, -1), hidden.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(self.adj_func(a_input)).squeeze(2)  # torch.matmul(a_input, self.a).squeeze(2))
        if adj is not None:
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e  # Batch x N x N
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, hidden)

        if self.activation:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, activation=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, activation=True)

    def forward(self, x, adj=None):
        # residual = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        # return F.log_softmax(x, dim=1)
        # return residual + x
        return x

# if __name__ == '__main__':
#     # layer = GraphAttentionLayer(256, 256, 0.4, 0.2).cuda()
#     # in_fea = torch.Tensor(1024, 256).cuda()
#     # adj = torch.ones(1024, 1024).cuda().detach()
#     # print(layer(in_fea, adj).shape)
#     gat = GAT(256, 256//8, 256, 0.4, 0.2, 8).cuda()
#     in_fea = torch.Tensor(1024, 256).cuda()
#     adj = torch.ones(1024, 1024).cuda().detach()
#     print(gat(in_fea, adj).shape)