
from dgl.nn.pytorch import GraphConv
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
class Self_Attention(nn.Module):
    def __init__(self,in_feats,d_k,d_v,device='cpu'):
        super(Self_Attention,self).__init__()
        self.W_Q = GraphConv(in_feats, d_k)
        self.W_K = GraphConv(in_feats, d_k)
        self.W_V = GraphConv(in_feats, d_v)
        self.W_O = GraphConv(d_v,in_feats)
        self.d_k=d_k
        self.device=device
    def forward(self,g,inputs,h_attn=None):
        print('input.shape',inputs.shape)
        Q = self.W_Q(g, inputs)
        K = self.W_K(g, inputs)
        V = self.W_V(g, inputs)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.FloatTensor([self.d_k])).to(self.device)
        if h_attn == None:
            attn = nn.Softmax(dim=-1)(scores)
        else:
            attn = nn.Softmax(dim=-1)(scores+h_attn)
        attn_out = torch.matmul(attn, V)
        attn_out=self.W_O(g,attn_out)
        return attn_out, attn
        pass

class GCNFeedforwardLayer(nn.Module):
    def __init__(self, in_feats, hidden_size,dropout):
        super(GCNFeedforwardLayer, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, in_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        out = self.dropout(torch.relu(self.conv1(g,inputs)))
        out=self.conv2(g,out)
        return out

class HCLLayer(nn.Module):
    def __init__(self,in_feats,d_k,d_v,hideen_size,dropout,device='cpu'):
        super(HCLLayer,self).__init__()
        self.in_feats=in_feats
        self.self_attention=Self_Attention(in_feats,d_k,d_v,device)
        self.ln=nn.LayerNorm(in_feats)
        self.feedforward=GCNFeedforwardLayer(in_feats,hideen_size,dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self,g,inputs,attn=None):
        if attn is not None:
            print('attn',attn.shape)
        attn_out,attn=self.self_attention(g,inputs,attn)
        attn_out=self.ln(attn_out)
        out=self.feedforward(g,attn_out+inputs)
        out=self.ln(out)
        return out,attn

'''
n_path_node:每条path有几个node（映射到几个node里面）
'''
class HCL(nn.Module):
    def __init__(self,n_path_node,n_layers,in_feats,d_k,d_v,hidden_size,dropout,num_class,device=''):
        super(HCL,self).__init__()
        self.device=device
        self.layers=nn.ModuleList([HCLLayer(in_feats,d_k,d_v,hidden_size,dropout,device) for _ in range(n_layers)])
        self.cla1 = nn.Linear(in_feats,128)
        self.cla2 = nn.Linear(128, num_class)
        self.n_path_node=n_path_node
        self.path = nn.ModuleList(nn.Linear(in_feats,in_feats) for _ in range(self.n_path_node))
        self.dropout = nn.Dropout(dropout)

    '''
    n_path_node 非类内的，当前经过的线性层的索引号
    '''
    def forward(self,g,node_emb,path_emb,path_node_dict=None,attn=None):
        for path in path_node_dict:
            n_path_node=0
            for node in path_node_dict[path]:
                if(self.n_path_node>n_path_node):
                    print(self.n_path_node)
                    node_emb[node] = node_emb[node] + self.path[n_path_node](path_emb[path])
                    n_path_node += 1

        for layer in self.layers:
            node_emb,attn =layer(g,node_emb,attn)
        fe = self.dropout(torch.relu(self.cla1(node_emb)))
        out=self.cla2(fe)
        return out,fe

    
def pooling(node_num,node_emb):
    node_begin = 0
    node=[]
    print('pooling,node_emb.shape',node_emb.shape)
    for i in range(len(node_num)):
        node_slice = node_emb[node_begin:node_begin + node_num[i]]
        node_begin=node_begin+node_num[i]
        print('node_slice',node_slice.shape)
        node.append(torch.sum(node_slice,0))
    ast_node= torch.stack(node)
    return ast_node


    
class ClaModel(nn.Module):
    def __init__(self,
                 in_feature,
                 n_path_node,
                 n_layers,
                 dropout,
                 n_class,
                 d_k=128, d_v=128, hidden_size=1024, device='cuda'):
        super(ClaModel, self).__init__()
        self.layers = nn.ModuleList([HCLLayer(in_feature, d_k, d_v, hidden_size, dropout, device) for _ in range(n_layers)])
        self.pooling = pooling
        self.cla = nn.Sequential(
            nn.Linear(in_feature, 1024),
                  nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_class),
        )

    def forward(self,g,node_emb,node_num,path_emb,path_node_dict,attn=None):
        for path in path_node_dict:
            n_path_node=0
            for node in path_node_dict[path]:
                if(self.n_path_node>n_path_node):
                    print(self.n_path_node)
                    node_emb[node] = node_emb[node] + self.path[n_path_node](path_emb[path])
                    n_path_node += 1

        for layer in self.layers:
            node_emb,attn =layer(g,node_emb,attn)

        ast=self.pooling(node_num,node_emb)
        print(ast.shape)
        cla_out=self.cla(ast)
        return cla_out


class CloModel(nn.Module):
    def __init__(self,
                 in_feature,
                 n_layers,
                 dropout,
                 n_class,
                 d_k=128, d_v=128, hidden_size=128, device='cuda'):
        super(CloModel, self).__init__()

        self.layers = nn.ModuleList(
            [HCLLayer(in_feature, d_k, d_v, hidden_size, dropout, device) for _ in range(n_layers)])
        # self.pre_model = pre_model
        self.pooling = pooling
        self.cla = nn.Sequential(
            nn.Linear(in_feature, 128),
            nn.ReLU(),
            nn.Linear(128, n_class),
        )
        '''
         def forward(self,g,node_emb,node_num,attn=None):
        for layer in self.layers:
            node_emb, attn = layer(g, node_emb, attn)
        ast=self.pooling(node_num,node_emb)
        cla_out=self.cla(ast)
        return cla_out
        '''

    def forward(self, g1, node_emb1, node_num1, path_emb1, path_node_dict1, g2, node_emb2, node_num2, path_emb2, path_node_dict2, attn=None):

        '''
        第一个code的node信息和path信息进行结合
        '''
        for path in path_node_dict1:
            n_path_node=0
            for node in path_node_dict1[path]:
                if(self.n_path_node>n_path_node):
                    node_emb1[node] = node_emb1[node] + self.path[n_path_node](path_emb1[path])
                    n_path_node += 1



        '''
        第二个code的node信息和path信息进行结合
         '''
        for path in path_node_dict2:
            n_path_node=0
            for node in path_node_dict2[path]:
                if(self.n_path_node>n_path_node):
                    node_emb2[node] = node_emb2[node] + self.path[n_path_node](path_emb2[path])
                    n_path_node += 1


        for layer in self.layers:
            node_emb1, attn = layer(g1, node_emb1, attn)
        ast1 = self.pooling(node_num1,node_emb1)

        attn = None
        for layer in self.layers:
            node_emb2, attn = layer(g2, node_emb2, attn)
        ast2 = self.pooling(node_num2,  node_emb2)

        ast_out = torch.abs(torch.add(ast1, -ast2))
        cla_out = self.cla(ast_out)
        cla_out = torch.sigmoid(cla_out)
        return cla_out



class Code2Vector(nn.Module):
    def __init__(self,
                 in_feats,
                 n_path_node,
                 n_layers,
                 dropout,
                 d_k=128, d_v=128, hidden_size=1024, device='cuda'):
        super(Code2Vector,self).__init__()
        self.device=device
        self.layers=nn.ModuleList([HCLLayer(in_feats,d_k,d_v,hidden_size,dropout,device) for _ in range(n_layers)])
        self.n_path_node=n_path_node
        self.pooling = pooling
        self.path = nn.ModuleList(nn.Linear(in_feats,in_feats) for _ in range(self.n_path_node))
    '''
    n_path_node 非类内的，当前经过的线性层的索引号
    '''
    def forward(self,g,node_emb,node_num,path_emb,path_node_dict,attn=None):
        for path in path_node_dict:
            n_path_node=0
            for node in path_node_dict[path]:
                if(self.n_path_node>n_path_node):
                    print(self.n_path_node)
                    node_emb[node] = node_emb[node] + self.path[n_path_node](path_emb[path])
                    n_path_node += 1

        for layer in self.layers:
            node_emb,attn =layer(g,node_emb,attn)
        ast=self.pooling(node_num,node_emb)
        return ast

#
# import dgl
# srcs = [0, 1, 2, 3, 4]
# dsts = [1, 0, 1, 4, 5]
# g = dgl.graph((srcs, dsts))
# g = dgl.add_self_loop(g)
# # g = g.to('cuda:0')
# #
# node_emb = torch.randn(6, 768)
# path_emb = torch.randn(3, 768)
# node_num = [4, 2]
# path_num = [1, 2]
# path_node_dict={0:[0,1,2],1:[0,1,3],2:[0,4,5]}
# # pre_model=torch.load('model.pkl')
# # pre_model_dict=pre_model.state_dict()
# model =HCL(2, 2, 768, 768, 768, 768, 0.5, 2, 'cpu')
# '''
# n_path_node,n_layers,in_feats,d_k,d_v,hidden_size,dropout,num_class,device
# g,node_emb,path_emb,path_node_dict=None,attn=None
# '''
#
# ast_out = model(g, node_emb, path_emb, path_node_dict,)