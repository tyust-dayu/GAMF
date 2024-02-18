import sys

sys.path.append("./../")
from torch.nn import LayerNorm, Linear, Dropout, Softmax
from einops import rearrange, repeat
import copy
from timm.models.layers import DropPath, trunc_normal_
from pathlib import Path
import re
import torch.backends.cudnn as cudnn
import record
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import math
from PIL import Image
import time
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
from scipy.io import loadmat as loadmat
from scipy import io
import torch.utils.data as dataf
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import einsum
import random
import numpy as np
import os
from layers_GCN import GraphConvolution
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from torch.nn import LayerNorm, Linear, Dropout, Softmax
import copy
import scipy.linalg
from torchvision import transforms as tfs

cudnn.deterministic = True
cudnn.benchmark = False


# %%
# GMF WITH CHANNEL TOKENIZATION


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
    # 对角矩阵 （负对角线上元素为0，其余为0） 维度扩充 重复 B*W


def distance(x):  # x是tensor digits返回的小数位数
    # 考虑加上节点自身 即再加一个对角矩阵
    # 需要加入batchsize 用多个小的邻接矩阵进行拼接为一个大的
    x_diag = torch.eye(x.shape[0], x.shape[0])
    distance = F.pdist(x)  # tensor 需要转格式使用squareform
    distance = distance.cpu().detach().numpy()  # 将tensor转为numpy
    distance = squareform(distance)
    distance = torch.from_numpy(distance) + x_diag
    return distance


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # 构建第一层 GCN
        # self.gc2 = GraphConvolution(nhid, nclass) # 构建第二层 GCN
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                           concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # 第二层的attention layer
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(HetConv, self).__init__()
        # groups 分组卷积的意思
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=g, padding=kernel_size // 3,
                             stride=stride)
        # Pointwise Convolution
        # self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.gwc(x)  # + self.pwc(x)


class IIEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(IIEmbedding, self).__init__()
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.pwc(x)


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 三个矩阵
        self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
        #         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        #         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        #         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        #         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        #         self.attn = Attention(dim = 64)
        self.attn = MCrossAttention(dim=dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)

        encoded = self.encoder_norm(x)

        return encoded[:, 0]


class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out + input


class ShareNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(ShareNet, self).__init__()
        # Pointwise Convolution
        self.share = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        hsi = self.share(x[0])
        iie = self.share(x[1])
        lidar = self.share(x[2])  # 把它堆成一个四维的
        out = torch.stack((hsi, iie, lidar))
        return out


class GAMF(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, HSIOnly):
        super(GAMF, self).__init__()
        self.HSIOnly = HSIOnly
        # HSI三维卷积 branch1
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 16, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(16, 32, (7, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(32, eps=1e-5),
            nn.GELU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(32, 64, (5, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(64, eps=1e-5),
            nn.GELU()
        )
        # HSI二维卷积
        self.conv8 = nn.Sequential(
            HetConv(2880, 512,
                    p=1,
                    g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                    ),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        # independent information embedding branch2
        self.conv9 = nn.Sequential(
            nn.Conv3d(1, 16, (9, 1, 1), padding=(0, 0, 0), stride=1),
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv3d(16, 32, (7, 1, 1), padding=(0, 0, 0), stride=1),
            nn.BatchNorm3d(32),
            nn.GELU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv3d(32, 64, (5, 1, 1), padding=(0, 0, 0), stride=1),
            nn.BatchNorm3d(64),
            nn.GELU()
        )
        self.IIEmbedding = nn.Sequential(
            IIEmbedding(2880, 512,
                        p=1,
                        g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                        ),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        # lidar二维卷积
        self.lidarConv1 = nn.Sequential(
            nn.Conv2d(NCLidar, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.lidarConv2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        self.share1 = nn.Sequential(
            ShareNet(512,
                     64,
                     p=1,
                     ),
            # nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.share2 = nn.Sequential(
            ShareNet(256,
                     64,
                     p=1,
                     ),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # 子注意力模块
        # self.att = selfattention(64)
        self.last_BandSize = NC // 2 // 2 // 2

        # 加入图卷积
        self.gcn = GCN(
            nfeat=64,
            nhid=64,
            nclass=64,
            dropout=0.6,
        )
        self.graph = GAT(nfeat=64,
                         nhid=8,
                         nclass=FM * 4,  # 最后的类别 相当于下一层的输入
                         dropout=0.6,
                         nheads=3,  # 之前是8个
                         alpha=0.2
                         )
        self.ca = TransformerEncoder(FM * 4)
        # Trento MUUFL n=4
        self.out3 = nn.Sequential(
            nn.Linear(448, 256),
            nn.ReLU()
        )
        # self.out3 = nn.Linear(FM * 20, 256)
        self.out4 = nn.Linear(256, Classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, 4 + 1, FM * 4))
        self.dropout = nn.Dropout(0.1)

        # 最后分类的两种初始化方式 初始化最后的权重 和 偏差
        # torch.nn.init.xavier_uniform_(self.out3.weight)
        # torch.nn.init.normal_(self.out3.bias, std=1e-6)

        # 标记化操作的两个参数 HSI和lidar各两个 并初始化
        # Trento Muufl 4 Houston 10
        self.token_wA = nn.Parameter(torch.empty(1, 6, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64),
                                       requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64),
                                       requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV_L)

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape) # HSIonly False torch.Size([4, 63, 121])  torch.Size([4, 1, 121])都在cuda上
        # HSIonly True torch.Size([4, 63, 121])在cuda上，x2 torch.Size([4, 1, 121]) 在CPU上
        x1 = x1.reshape(x1.shape[0], -1, patchsize, patchsize)  # 行数 个数/行数 torch.Size([4, 63, 11, 11])
        x1 = x1.unsqueeze(1)  # HSI在第二维上增加一个维度 [4, 1, 63, 11, 11]
        x_IIE = x1

        # 开始卷积 branch1
        x1 = self.conv5(x1)  # 三维卷积 [4, 8, 55, 11, 11]
        x1 = self.conv6(x1)  # [1, 16, 47, 11, 11]
        x1 = self.conv7(x1)
        x1 = x1.reshape(x1.shape[0], -1, patchsize, patchsize)  # [4, 1504, 11, 11]
        x1 = self.conv8(x1)  # HETconv [4, 64, 11, 11]

        # branch2
        x_IIE = self.conv9(x_IIE)  # [4, 8, 55, 11, 11]
        x_IIE = self.conv10(x_IIE)
        x_IIE = self.conv11(x_IIE)
        x_IIE = x_IIE.reshape(x1.shape[0], -1, patchsize, patchsize)  # torch.Size([4, 1504, 11, 11])
        x_IIE = self.IIEmbedding(x_IIE)  # torch.Size([4, 64, 11, 11])

        # branch3
        x2 = x2.reshape(x2.shape[0], -1, patchsize, patchsize)  # LIDAR torch.Size([4, 1, 11, 11]) cuda
        x2 = self.lidarConv1(x2)  # [4, 64, 11, 11]
        x2 = self.lidarConv2(x2)
        # x2 = self.att(x2)

        # 参数共享
        x = torch.stack((x1, x_IIE, x2))
        out = self.share1(x)
        x1, x_IIE, x2 = out[0], out[1], out[2]
        # x = torch.stack((x1, x_IIE, x2))
        # out = self.share2(x)
        # x1, x_IIE, x2 = out[0].unsqueeze(0), out[1].unsqueeze(0), out[2].unsqueeze(0)

        x1 = x1 + x_IIE

        # Tokenization
        x2 = x2.reshape(x2.shape[0], -1, patchsize ** 2)  # [4, 64, 121]
        x2 = x2.transpose(-1, -2)  # [4, 121, 64]

        wa_L = self.token_wA_L.expand(x1.shape[0], -1, -1)  # [4,1,64] cuda
        wa_L = rearrange(wa_L, 'b h w -> b w h')  # Transpose [4 64 1]
        A_L = torch.einsum('bij,bjk->bik', x2, wa_L)  # [4 121 1]
        A_L = rearrange(A_L, 'b h w -> b w h')  # Transpose [4 1 121]
        A_L = A_L.softmax(dim=-1)  # [4 1 121]
        wv_L = self.token_wV_L.expand(x2.shape[0], -1, -1)
        VV_L = torch.einsum('bij,bjk->bik', x2, wv_L)
        x2 = torch.einsum('bij,bjk->bik', A_L, VV_L)  # [4 1 64]

        x1 = x1.flatten(2)  # [4 64 121 ]
        x1 = x1.transpose(-1, -2)  # [4 121 64 ]
        wa = self.token_wA.expand(x1.shape[0], -1, -1)  # [4, 4, 64])
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose[4, 64, 4]
        A = torch.einsum('bij,bjk->bik', x1, wa)  # 121, 64 * 64, 10  -->121, 10
        A = rearrange(A, 'b h w -> b w h')  # Transpose [4, 10, 121]
        A = A.softmax(dim=-1)  # [4, 10, 121]
        wv = self.token_wV.expand(x1.shape[0], -1, -1)  # [4, 64, 64]
        VV = torch.einsum('bij,bjk->bik', x1, wv)  # [4, 121, 64])
        T = torch.einsum('bij,bjk->bik', A, VV)  # 10, 64
        x = torch.cat((x2, T), dim=1)  # x2 [4, 1, 64] T [4, 4, 64] [b,n+1,dim] [batchsize, 5, 64]
        # embeddings = x + self.position_embeddings  # 位置编码没用
        # embeddings = self.dropout(embeddings)  # tensor [batchsize， 5， 64]

        # 在这儿加入图 对每一个HSI和其对应的LIDAR进行构图
        batch_adjust = []
        # 没有加入位置编码
        for i in range(x.shape[0]):
            batch_adjust = scipy.linalg.block_diag(batch_adjust, distance(x[i]))  # .cpu().detach().numpy()
            # distance(x[i]) # tensor切片
        batch_adjust = torch.tensor(batch_adjust[1:, :])  # [20, 20] cpu
        x = x.reshape(x.shape[0] * x.shape[1], -1)  # [20, 64]
        # adjust = distance(batch_distance)  #  [20 20 ] 计算结点间的距离矩阵 之后进行构图 然后就是两层的GAN  边有了
        # 每个节点的特征，邻接矩阵
        # x = self.gcn(x, batch_adjust.cuda())

        x = self.graph(x, batch_adjust.cuda())  # [20 64]  houston 44, 64

        # x = self.ca(x)  # x 应该是embeddings x包括两部分 是一个list 【2， 64】
        # x = x.reshape(x.shape[0], -1) # [10 64]

        # trento Muufl 使用 Houston最后设为n=10，x [44 64]对out3进行修改 [4 704]
        x = x.reshape(x1.shape[0], -1)
        out3 = self.out3(x)  # batchsize classes
        out4 = self.out4(out3)
        return out4


DATASETS_WITH_HSI_PARTS = ['Berlin', 'Augsburg']
DATA2_List = ['SAR', 'DSM', 'MS']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# All datasets = "Houston","Trento","MUUFL","HoustonMS","AugsburgSAR","AugsburgDSM"
datasetNames = ["Trento"]

# houston 两处lidar数据扩充维度

patchsize = 11
batchsize = 8  # 原始32 Trento 4可以跑
testSizeNumber = 100  # 500 分组检验模型的准确率，只有六类，不需要训练集很大
EPOCH = 0
BandSize = 1
LR = 5e-4
FM = 16  # ???
HSIOnly = False
FileName = 'GAMF'


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(xtest, xtest2, ytest, name, model):
    pred_y = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // testSizeNumber  # 819 // 500 = 2
    for i in range(number):
        temp = xtest[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
        temp = temp.cuda()

        temp1 = xtest2[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
        temp1 = temp1.cuda()

        temp2 = model(temp, temp1)

        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
        del temp, temp2, temp3, temp1

    if (i + 1) * testSizeNumber < len(ytest):
        temp = xtest[(i + 1) * testSizeNumber:len(ytest), :, :]
        temp = temp.cuda()
        temp1 = xtest2[(i + 1) * testSizeNumber:len(ytest), :, :]
        temp1 = temp1.cuda()

        temp2 = model(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * testSizeNumber:len(ytest)] = temp3.cpu()
        del temp, temp2, temp3, temp1

    pred_y = torch.from_numpy(pred_y).long()

    if name == 'Houston':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass'
            , 'Trees', 'Soil', 'Water',
                        'Residential', 'Commercial', 'Road', 'Highway',
                        'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                        'Running Track']
    elif name == 'Trento':
        target_names = ['Apples', 'Buildings', 'Ground', 'Woods', 'Vineyard',
                        'Roads']
    elif name == 'MUUFL' or name == 'MUUFLS' or name == 'MUUFLSR':
        target_names = ['Trees', 'Grass_Pure', 'Grass_Groundsurface', 'Dirt_And_Sand', 'Road_Materials', 'Water',
                        'Buildings', "Buildings'_Shadow",
                        'Sidewalk', 'Yellow_Curb', 'ClothPanels']  # 需要更改
    elif name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'UP':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

    #     classification = classification_report(ytest, pred_y, target_names=target_names)
    oa = accuracy_score(ytest, pred_y)
    confusion = confusion_matrix(ytest, pred_y)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(ytest, pred_y)

    return confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_tf(x):
    """
    transforms.CenterCrop 对图片中心进行裁剪
    transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
    transforms.FiveCrop  对图像四个角和中心进行裁剪得到五分图像
    transforms.Grayscale  对图像进行灰度变换
    transforms.Pad  使用固定值进行像素填充
    transforms.RandomAffine  随机仿射变换
    transforms.RandomCrop  随机区域裁剪
    transforms.RandomHorizontalFlip  随机水平翻转
    transforms.RandomRotation  随机旋转
    transforms.RandomVerticalFlip  随机垂直翻转
    :param x:
    :return:
    """
    im_aug = tfs.Compose([
        tfs.RandomHorizontalFlip(),  # 随机竖直翻转
        tfs.RandomRotation(90),  # 随机90度旋转
        tfs.RandomVerticalFlip(),
        tfs.ToTensor(),
    ])
    x = im_aug(x)
    return x


for BandSize in [1]:
    for datasetName in datasetNames:
        print("----------------------------------Training for ", datasetName,
              " ---------------------------------------------")
        try:
            os.makedirs(datasetName)
        except FileExistsError:
            pass
        data1Name = ''
        data2Name = ''
        # 只用这三个数据集吧
        if datasetName in ["Houston", "Trento", "MUUFL"]:
            data1Name = datasetName
            data2Name = "LIDAR"
        else:
            for dataName in DATA2_List:
                dataNameToCheck = re.compile(dataName)
                matchObj = dataNameToCheck.search(datasetName)
                if matchObj:
                    data1Name = datasetName.replace(dataName, "")
                    data2Name = dataName

        HSI = io.loadmat('./' + data1Name + '11x11/HSI_Tr.mat')  # dict 训练集
        TrainPatch = HSI['Data']  # numpy.ndarray
        TrainPatch = TrainPatch.astype(np.float32)
        NC = TrainPatch.shape[3]  # NC is number of bands

        LIDAR = io.loadmat('./' + data1Name + '11x11/' + data2Name + '_Tr.mat')  # (819, 11, 11, 1)
        TrainPatch2 = LIDAR['Data']  # 把字典类型转为numpy

        # houston 在最后加一维 (2832, 11, 11, 1)
        # TrainPatch2 = np.expand_dims(TrainPatch2, 3)

        TrainPatch2 = TrainPatch2.astype(np.float32)  # dtype('float32')
        NCLIDAR = TrainPatch2.shape[3]  # NCLIDAR is number of bands

        label = io.loadmat('./' + data1Name + '11x11/TrLabel.mat')
        TrLabel = label['Data']

        # Test data
        # DATASETS_WITH_HSI_PARTS = ['Berlin', 'Augsburg'] 不使用这两个数据集
        if data1Name in DATASETS_WITH_HSI_PARTS:
            i = 2
            basePath = "./" + data1Name + '11x11/HSI_Te_Part'
            TestPatch = io.loadmat(basePath + str(i - 1) + '.mat')['Data']
            while True:
                my_file = Path(basePath + str(i) + '.mat')
                if my_file.exists():
                    TestPatch = np.concatenate([TestPatch, io.loadmat(basePath + str(i) + '.mat')['Data']], axis=0)
                    i += 1
                else:
                    break
        else:
            HSI = io.loadmat('./' + data1Name + '11x11/HSI_Te.mat')
            TestPatch = HSI['Data']

        # ["Houston", "Trento", "MUUFL"]: 测试集数据
        TestPatch = TestPatch.astype(np.float32)  # test HSI

        LIDAR = io.loadmat('./' + data1Name + '11x11/' + data2Name + '_Te.mat')  # test LIDAR
        TestPatch2 = LIDAR['Data']
        TestPatch2 = TestPatch2.astype(np.float32)

        label = io.loadmat('./' + data1Name + '11x11/TeLabel.mat')  # test label
        TsLabel = label['Data']

        TrainPatch1 = torch.from_numpy(TrainPatch).to(torch.float32)
        TrainPatch1 = TrainPatch1.permute(0, 3, 1, 2)  # 维度换位 819 11 11 63 -》819 63 11 11 HSI
        TrainPatch1 = TrainPatch1.reshape(TrainPatch1.shape[0], TrainPatch1.shape[1], -1).to(
            torch.float32)  # 819 63 121
        TrainPatch2 = torch.from_numpy(TrainPatch2).to(torch.float32)  # LIDAR
        TrainPatch2 = TrainPatch2.permute(0, 3, 1, 2)
        TrainPatch2 = TrainPatch2.reshape(TrainPatch2.shape[0], TrainPatch2.shape[1], -1).to(
            torch.float32)  # 819 63 121
        TrainLabel1 = torch.from_numpy(TrLabel) - 1  # mat里面标签为1 到 6
        TrainLabel1 = TrainLabel1.long()
        TrainLabel1 = TrainLabel1.reshape(-1)

        TestPatch1 = torch.from_numpy(TestPatch).to(torch.float32)  # test HSI
        TestPatch1 = TestPatch1.permute(0, 3, 1, 2)
        TestPatch1 = TestPatch1.reshape(TestPatch1.shape[0], TestPatch1.shape[1], -1).to(torch.float32)
        TestPatch2 = torch.from_numpy(TestPatch2).to(torch.float32)  # test LIDAR

        # houston 在最后加一维 (2832, 11, 11, 1) houston数据新加
        # TestPatch2 = TestPatch2.unsqueeze(dim=3)

        TestPatch2 = TestPatch2.permute(0, 3, 1, 2)
        TestPatch2 = TestPatch2.reshape(TestPatch2.shape[0], TestPatch2.shape[1], -1).to(torch.float32)
        TestLabel1 = torch.from_numpy(TsLabel) - 1
        TestLabel1 = TestLabel1.long()
        TestLabel1 = TestLabel1.reshape(-1)

        Classes = len(np.unique(TrainLabel1))
        dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel1)  # HSI LIDAR label是HSI个label
        # 不考虑Berlin数据集
        if data1Name in ['Berlin']:
            train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0)
        else:
            train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0)

        print("HSI Train data shape = ", TrainPatch1.shape)
        print(data2Name + " Train data shape = ", TrainPatch2.shape)
        print("Train label shape = ", TrainLabel1.shape)

        print("HSI Test data shape = ", TestPatch1.shape)
        print(data2Name + " Test data shape = ", TestPatch2.shape)
        print("Test label shape = ", TestLabel1.shape)

        print("Number of Classes = ", Classes)
        KAPPA = []
        OA = []
        AA = []
        ELEMENT_ACC = np.zeros((3, Classes))

        set_seed(42)
        for iterNum in range(3): # train 3, test 1
            # NC HSI波段数 NCLidar LIDAR波段数
            model = GAMF(FM, NC, NCLIDAR, Classes, HSIOnly).cuda()
            # summary(model, [(NC, patchsize ** 2), (NCLIDAR, patchsize ** 2)]) # 估计模型参数大小 model inputsize

            log_dir = "./Trento/net_params_GAMF.pkl"
            if os.path.exists(log_dir):
                checkpoint = torch.load(log_dir)
                # for k, v in checkpoint.items():
                #     print(k)
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-3)
            loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
            BestAcc = 0
            accuracy = 0
            # torch.cuda.synchronize() # 测试时间同步
            start = time.time()
            # train and test the designed model
            for epoch in range(EPOCH):
                # train_loader = dataf.DataLoader(data, batch_size=batchsize, shuffle=True, num_workers=0)
                # data = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel1)
                for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

                    # move train data to GPU
                    b_x1 = b_x1.cuda()
                    b_y = b_y.cuda()
                    if HSIOnly:  # 不看
                        out1 = model(b_x1, b_x2)
                        loss = loss_func(out1, b_y)
                    else:
                        b_x2 = b_x2.cuda()
                        out = model(b_x1, b_x2)
                        loss = loss_func(out, b_y)
                    # if accuracy > 0.95:
                    #     # LR = 5e-4
                    #     optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-3)

                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    if step % 50 == 0:
                        model.eval()
                        pred_y = np.empty((len(TestLabel1)), dtype='float32')
                        number = len(TestLabel1) // testSizeNumber
                        for i in range(number):
                            temp = TestPatch1[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                            temp = temp.cuda()
                            temp1 = TestPatch2[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                            temp1 = temp1.cuda()
                            if HSIOnly:
                                temp2 = model(temp, temp1)
                                temp3 = torch.max(temp2, 1)[1].squeeze()
                                pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cuda()
                                del temp, temp2, temp3
                            else:
                                temp2 = model(temp, temp1)  # HSI LIDAR
                                temp3 = torch.max(temp2, 1)[1].squeeze()
                                pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cuda()
                                del temp, temp1, temp2, temp3

                        if (i + 1) * testSizeNumber < len(TestLabel1):
                            temp = TestPatch1[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                            temp = temp.cuda()
                            temp1 = TestPatch2[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                            temp1 = temp1.cuda()
                            if HSIOnly:
                                temp2 = model(temp, temp1)
                                temp3 = torch.max(temp2, 1)[1].squeeze()
                                pred_y[(i + 1) * testSizeNumber:len(TestLabel1)] = temp3.cuda()
                                del temp, temp2, temp3
                            else:
                                temp2 = model(temp, temp1)
                                temp3 = torch.max(temp2, 1)[1].squeeze()
                                pred_y[(i + 1) * testSizeNumber:len(TestLabel1)] = temp3.cuda()
                                del temp, temp1, temp2, temp3

                        pred_y = torch.from_numpy(pred_y).long()
                        accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

                        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cuda().numpy(),
                              '| test accuracy: %.4f' % (accuracy * 100))

                        # save the parameters in network
                        if accuracy > BestAcc:
                            BestAcc = accuracy

                            torch.save(model.state_dict(), datasetName + '/net_params_' + FileName + '.pkl')

                        model.train()
                scheduler.step()
            # torch.cuda.synchronize()
            end = time.time()
            print(end - start)
            Train_time = end - start

            # load the saved parameters

            model.load_state_dict(torch.load(datasetName + '/net_params_' + FileName + '.pkl'))

            model.eval()
            confusion, oa, each_acc, aa, kappa = reports(TestPatch1, TestPatch2, TestLabel1, datasetName, model)
            KAPPA.append(kappa)
            OA.append(oa)
            AA.append(aa)
            ELEMENT_ACC[iterNum, :] = each_acc
            torch.save(model, datasetName + '/best_model_' + FileName + '_BandSize' + str(BandSize) + '_Iter' + str(
                iterNum) + '.pt')

            print("OA = ", oa)
        print("----------" + datasetName + " Training Finished -----------")
        record.record_output(OA, AA, KAPPA, ELEMENT_ACC, './' + datasetName + '/' + FileName + '_BandSize' + str(
            BandSize) + '_Report_' + datasetName + '.txt')

# %%
# model = GMF(FM, NC, NCLIDAR, Classes, HSIOnly).cuda()
# model = (16, 144, 1, 6, HSIOnly) # False
# summary(model, [(144, 121), (1, 121)], device='cuda')  # , depth=5
