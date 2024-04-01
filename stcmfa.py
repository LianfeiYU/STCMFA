
import torch
import torch.nn as nn
import torch.nn.init as init

import math
import torch.nn.functional as F

from flownet import FlowClassiregressor
from flow_attention import AttentionLayer, Flow_Attention

class ChebConv(nn.Module):# 定义图卷积层的类
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize # 正则化参数,True or False

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c] ,第二个1是维度扩张，计算方便,有没有都不影响参数的大小,nn.Parameter就是把参数转换成模型可改动的参数.
        # 之所以要k+1,是因为k是从0开始的
        init.xavier_normal_(self.weight)  # 用正态分布填充

        if bias: # 偏置,就是一次函数中的b
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))  # 前面的两个1是为了计算简单，因为输出的维度是3维
            init.zeros_(self.bias)  # 用0填充
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N],得到拉普拉斯矩阵
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]，这个就是多阶的切比雪夫多项式，K就是阶数，N是节点数量

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]，这个就是计算完后乘x
        result = torch.matmul(result, self.weight)  # [K, B, N, D]，计算上一步之后乘W

        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]，求和

        return result

    def cheb_polynomial(self, laplacian): # 计算切比雪夫多项式,也就是前面公式中的 T_k(L)
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N] ,节点个数
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N],初始化一个全0的多项式拉普拉斯矩阵
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)  # 0阶的切比雪夫多项式为单位阵

        if self.K == 1: # 这个self.k就是前面说的0阶切比雪夫多项式
            return multi_order_laplacian
        else: # 大于等于1阶
            multi_order_laplacian[1] = laplacian
            if self.K == 2: # 1阶切比雪夫多项式就是拉普拉斯矩阵 L 本身
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2] #切比雪夫多项式的递推式:T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize): # 计算拉普拉斯矩阵
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2)) # 这里的graph就是邻接矩阵,这个D
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D) # L = I - D * A * D,这个也就是正则化
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class STCMFA(nn.Module):  # 定义图网络的类
    def __init__(self, in_c, hid_c, out_c, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.class
        :param out_c: int, number of output channels.
        :param K:
        """
        super(STCMFA, self).__init__()

        self.linear_1 = nn.Linear(12, 1024)  # 定义一个线性层

        self.conv1 = ChebConv(in_c=1024, out_c=512, K=K)  # 第1个图卷积层

        self.conv = ChebConv(in_c=512, out_c=256, K=K)  # 第2个图卷积层

        self.conv2 = ChebConv(in_c=256, out_c=128, K=K)  # 第3个图卷积层

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)

        self.maxpool2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)

        self.fla = FlowClassiregressor(feat_dim = 64, max_len=10000)

        self.linear_2 = nn.Linear(64, 24)  # 定义一个线性层

        self.act_relu = nn.ReLU()
        # self.act_tanh = nn.Tanh()
        # self.act_sig = nn.Sigmoid()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        # print(graph_data.shape)
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]  # B是batch size，N是节点数，H是历史数据长度，D是特征维度

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 6, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起

        # 添加线性层
        # output_1 = self.act(self.conv1(output_1, graph_data))
        flow_x = self.linear_1(flow_x)

        output_1 = self.act_relu(self.conv1(flow_x, graph_data))         #conv1
        # output_1 = self.dropout(output_1)
        # print(output_1.size())
        # output_1, _ = self.gru(output_1)
        # output_1 = self.maxpool2(output_1)
        output_1 = self.act_relu(self.conv(output_1, graph_data))           #conv2
        output_1 = self.act_relu(self.conv2(output_1, graph_data))          #conv3
        # output_1 = self.rh(flow_x, graph_data, self.node_embeddings)
        # output_1 = self.act_relu(self.conv2(output_1, graph_data))
        # output_1 = self.act_relu(self.conv3(output_1, graph_data))
        output_1 = self.maxpool1(output_1)
        # i = output_1

        # output_1 = self.att(output_1)
        output_1 = self.fla(output_1)
        # output_1 = self.fla(output_1, output_1,output_1)
        # output_1 = output_1.permute(0, 2, 1)
        # output_1 = self.nn_conv1(output_1)
        # output_1 = output_1.permute(0, 2, 1)
        # output_1, _ = self.gru(output_1)
        output_1 = self.linear_2(output_1)
        # output_1 = self.act_relu(output_1)
        # output_1, _ = self.gru2(output_1)
        output_1 = self.maxpool2(output_1)

        return output_1.unsqueeze(3)  # 在第 3 维度，也就是时间维度上做扩张