import javalang
from javalang.ast import Node
from tqdm import tqdm
import torch
from Node import ASTNode, SingleNode
from pycparser import c_parser
import copy
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings
import pandas as pd
import numpy as np
# init embedding
embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

class DataSet :
    '''
    输入：文件路径，具体到具体路径
    返回：node path列表，path和node对应关系，src，dst,label
    '''
    def __init__(self, data_path=''):
        self.pathID = 0
        self.ID = 0
        self.node,self.path,self.label,self.src,self.dst,self.path_node_list,self.n_ast_node=[],[],[],[],[],[],[]
        self.path_dict,self.node_dict,self.p_n_dict=dict(),dict(),dict()

    def reset(self):
        self.pathID = 0
        self.ID = 0
        self.node, self.path, self.label, self.src, self.dst, self.path_node_list, self.n_ast_node = [], [], [], [], [], [], []
        self.path_dict, self.node_dict, self.p_n_dict = dict(), dict(), dict()

    def delete_blank(str):
        strRes = str.replace(' ', '').replace('\n', '').replace('\r', '')
        return strRes

    def get_emb(str_):
        if type(str_) == list:
            str_ = ''.join(str_)
        sen = DataSet.delete_blank(str_)
        sentence = Sentence(sen)
        with torch.no_grad():
            torch.cuda.empty_cache()
            embedding.embed(sentence)
        emb = sentence.embedding.detach().cpu().numpy().tolist()
        return emb

    def get_bcb_ast(self,code):
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def java_get_children(root):
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def java_get_token(node):
        token = ''
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):
            token = 'Modifier'  # node.pop()
        elif isinstance(node, Node):
            token = node.__class__.__name__
        return token

    '''
    get_node需要进行的操作
    记住自己的id
    
    '''
    def java_get_node(self,node,deep_label, pathID_list):
        # global ID, path_dict, node_dict, p_n_dict
        id = self.ID
        token, children = DataSet.java_get_token(node), DataSet.java_get_children(node)
        self.path_node_list.append(token)
        pathID_list.append(id)
        node_content = ''
        if isinstance(node, Node):
            node_content= node_content + str(node.position)
        if len(children) == 0:
            node_content = node_content + str(token)
            self.path_dict[self.pathID] = copy.deepcopy(self.path_node_list)
            self.p_n_dict[self.pathID] = copy.deepcopy(pathID_list)
            self.path.append(copy.deepcopy(DataSet.get_emb(self.path_node_list)))
            self.pathID += 1
        else:
            node_content = node_content + str(token) + str(children)

        self.node.append(DataSet.get_emb(node_content))
        self.label.append(deep_label)

        for child in children:
            if DataSet.java_get_token(child) == '':
                continue
            self.ID += 1
            self.src.append(id)
            self.dst.append(self.ID)
            DataSet.java_get_node(self,child, deep_label + 1,  pathID_list)
            self.path_node_list.pop()
            pathID_list.pop()

    def get_gcj_ast(self,code):
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse()
        return tree

    def parse_java(self, data, dataset_name='gcj'):
        DataSet.reset(self)
        pathID_list = []
        gcj_class_label = []
        for i, j in data.iterrows():
            if dataset_name == 'gcj':
                ast = DataSet.get_gcj_ast(self, j['code'])
                gcj_class_label.append(j['file_label'])
            elif dataset_name == 'bcb':
                ast = DataSet.get_bcb_ast(self, j[1])
            else:
                raise Exception('The dataset name is wrong.')
            DataSet.java_get_node(self, ast, 0, pathID_list)
            if len(self.n_ast_node) == 0:
                self.n_ast_node.append(len(self.label))
            else:
                self.n_ast_node.append(len(self.label) - self.n_ast_node[-1])
        return self.node, self.path, self.label, self.p_n_dict, self.n_ast_node, self.src, self.dst, gcj_class_label

    def c_get_ast(self,code):
        parser = c_parser.CParser()
        ast = parser.parse(code)
        return ast

    def c_get_node(self,node,deep_label,pathID_list):
        id = self.ID
        node_content=str(node.coord) + DataSet.delete_blank(str(node))
        pathID_list.append(id)
        '''
        获取到相关信息之后直接进行编码然后进行保存
        '''
        emb=DataSet.get_emb(node_content)
        self.node.append(emb)
        self.label.append(deep_label)
        self.path_node_list.append(type(node))
        if node.children() is not None:
            for x, y in node.children():
                self.ID += 1
                self.src.append(id)
                self.dst.append(self.ID)
                DataSet.c_get_node(self, y, deep_label + 1, pathID_list)
                self.path_node_list.pop()
                pathID_list.pop()
        else:
            self.path_dict[self.pathID]=copy.deepcopy(self.path_node_list)
            self.p_n_dict[self.pathID]=copy.deepcopy(pathID_list)
            self.path.append(copy.deepcopy(DataSet.get_emb(self.path_node_list)))
            self.pathID += 1

    def parse_c(self, data, dataset_name='oj'):
        DataSet.reset(self)
        pathID_list = []
        oj_class_label = []

        for i, j in data.iterrows():
            begin_temp = self.ID
            if dataset_name == 'oj':
                ast = DataSet.c_get_ast(self, j[1])
                oj_class_label.append(j[2]-1)
            else:
                raise Exception('The dataset name is wrong.')
            DataSet.c_get_node(self, ast, 0, pathID_list)
            if len(self.n_ast_node) == 0:
                self.n_ast_node.append(len(self.label))
            else:
                self.n_ast_node.append(len(self.label) - self.n_ast_node[-1])
            end_temp = self.ID - begin_temp
            self.ID += 1
        #     print('begin,end,all',begin_temp,end_temp,end_temp-begin_temp)
        # print('n_parse_code,187,len:(node,src,dst)',len(self.node),len(self.label),len(self.src),len(self.dst))
        return self.node, self.path, self.label, self.p_n_dict, self.n_ast_node, self.src, self.dst, oj_class_label


def datasetSplit(data_path,train_ratio,valid_ratio,test_ratio,data_suf='csv'):
    '''
    用来对数据集进行随机化，并按照比例进行分割
    返回处理好的，训练数据集，验证数据集，和测试数据集
    '''
    if data_suf == 'csv':
        data = pd.read_csv(data_path, sep='\t', header=0, encoding='utf-8')
    if data_suf == 'pkl':
        data = pd.read_pickle(data_path)

    data_len = len(data)
    train_id,valid_id,test_id=[],[],[]
    import random
    sample_id = random.sample(range(0, data_len), data_len)
    train_split = int(data_len * train_ratio * 0.1)
    valid_split = int(data_len * train_ratio * 0.1 + data_len * valid_ratio * 0.1)
    train_id = sample_id[0 : train_split]
    valid_id = sample_id[ train_split : valid_split]
    test_id = sample_id[valid_split: ]
    train_data = data.iloc[train_id]
    valid_data = data.iloc[valid_id]
    test_data = data.iloc[test_id]
    return train_data,valid_data,test_data



def clo_datasetSplit(pair_path,data_path,train_ratio,valid_ratio,test_ratio):
    '''
    用来对数据集进行随机化，并按照比例进行分割
    返回处理好的，训练数据集，验证数据集，和测试数据集
    '''
    pair = pd.read_pickle(pair_path)
    data =  pd.read_pickle(data_path)

    data_len = len(pair)
    train_id,valid_id,test_id=[],[],[]
    import random
    sample_id = random.sample(range(0, data_len), data_len)
    train_split = int(data_len * train_ratio * 0.1)
    valid_split = int(data_len * train_ratio * 0.1 + data_len * valid_ratio * 0.1)
    train_id = sample_id[0 : train_split]
    valid_id = sample_id[train_split : valid_split]
    test_id = sample_id[valid_split: ]


    train_data_id1 = pair.iloc[train_id,0]
    train_data_id2 = pair.iloc[train_id,1]
    train_data_label = pair.iloc[train_id,2].tolist()

    valid_data_id1 = pair.iloc[valid_id, 0]
    valid_data_id2 = pair.iloc[valid_id, 1]
    valid_data_label = pair.iloc[valid_id, 2].tolist()

    test_data_id1 = pair.iloc[test_id, 0]
    test_data_id2 = pair.iloc[test_id, 1]
    test_data_label = pair.iloc[test_id, 2].tolist()

    train_data1 = data.iloc[train_data_id1]
    train_data2 = data.iloc[train_data_id2]
    valid_data1 = data.iloc[valid_data_id1]
    valid_data2 = data.iloc[valid_data_id2]

    test_data1 = data.iloc[test_data_id1]
    test_data2 = data.iloc[test_data_id2]

    return train_data1, train_data2, train_data_label, valid_data1, valid_data2, valid_data_label, test_data1, test_data2, test_data_label




def getMaxDeep(data_path,maxDeep,dataSetName='oj'):
    dataset=DataSet()
    data=pd.read_pickle(data_path)
    if dataSetName=='oj':
        def c_get_node(node, deep_label,maxDeep):
            if deep_label > maxDeep[0]:
                maxDeep[0] = deep_label
            if node.children() is not None:
                for x,y in node.children():
                    c_get_node(y,deep_label+1,maxDeep)


        for i, j in tqdm(data.iterrows()):
            ast = dataset.c_get_ast(j[1])
            c_get_node(ast,0,maxDeep)



    elif dataSetName=='gcj':
        def java_get_node(node, deep_label,maxDeep):
            token, children = DataSet.java_get_token(node), DataSet.java_get_children(node)
            if len(children) == 0:
                if deep_label > maxDeep[0]:
                    maxDeep[0] = deep_label
                    print(maxDeep)
            for child in children:
                java_get_node(child,deep_label+1,maxDeep)

        for i, j in data.iterrows():
            ast = dataset.get_gcj_ast(j['code'])
            java_get_node(ast,0,maxDeep)




datapath='data/gcj/test.pkl'
maxDeep = [1]
getMaxDeep(datapath,maxDeep,'gcj')
print(maxDeep[0])
# # data=pd.read_csv('data/bcb/bcb_funcs.tsv', sep='\t', header=0, encoding='utf-8')
# # data=pd.read_pickle('data/gcj/test.pkl')
# data=pd.read_pickle('data/oj/programs.pkl')
# G=DataSet()
# '''
# 解析了data中前4行数据
# '''
# id_list = [36718, 1646]
# data1=data.iloc[id_list]
# a,b,c,d,e,f,g,f=G.parse_c(data1,'oj')

#
# pair_path = 'data/clone_pair/oj_clone_ids.pkl'
# data_path = 'data/oj/programs.pkl'
# clo_datasetSplit(pair_path,data_path,8,1,1)


# data=pd.read_pickle('data/oj/programs.pkl')
# print(data)
# label=[]
# for i ,j in data.iterrows():
#     label.append(j[2])
# print(max(label),min(label))


