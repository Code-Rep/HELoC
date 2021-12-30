from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from n_model import Code2Vector
import dgl
import torch
import time
from n_parse_code import datasetSplit,DataSet
from sklearn import metrics
from numpy import *



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--data_path', default='data/oj/programs.pkl',
                        help='node emb path')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--pre_model', default='model.pkl',
                        help='pre_model')
    args = parser.parse_args()
    data_path = args.data_path
    device = args.device
    pre_model = args.pre_model

    '''
    根据数据集需要更改的信息
    '''
    dataset_name = 'oj'
    n_clusters = 104

    train_ratio, val_ratio, test_ratio = 8, 1, 1
    train_data,val_data,test_data = datasetSplit(data_path,train_ratio,val_ratio,test_ratio,data_suf='pkl')
    dataSet = DataSet()

    begin = 0
    '''
    定义一些超参数
    '''
    lr, BATCH_SIZE, EPOCH = 0.00001,128, 150
    in_feats,n_layer,drop_out=768,2,0.5
    n_path_node = 13

    '''
    对模型进行定义
    '''
    model = Code2Vector(in_feats,n_path_node,n_layer,drop_out, device = device).to(device)
    pre_model_dict = torch.load(pre_model)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    kmeans = MiniBatchKMeans(n_clusters = n_clusters,
                             random_state=0,
                             batch_size=BATCH_SIZE)

    pbar = tqdm(range(EPOCH))
    ARI = []
    for epoch in pbar:
        pbar.set_description('epoch:%d  processing' % (epoch))
        i = 0
        start_time = time.time()
        total_ari = 0.0

        while (i + BATCH_SIZE) <= len(train_data):
            input_node, path, label, p_n_dict, n_ast_node, src, dst, downtown_label=dataSet.parse_c(train_data.iloc[i:i+BATCH_SIZE,:], dataset_name)
            i = i + BATCH_SIZE

            '''
            获取模型的输入
            '''

            g = dgl.graph((src, dst))
            g = dgl.add_self_loop(g)
            g = g.to(device)
            input_node = torch.FloatTensor(input_node).to(device)
            path = torch.FloatTensor(path).to(device)

            logits = model(g,input_node, n_ast_node,path, p_n_dict).detach().numpy()
            print(logits.shape)
            kmeans = kmeans.partial_fit(logits)

            labels_pred = kmeans.predict(logits)
            ARI.append(metrics.adjusted_rand_score(downtown_label, labels_pred))
        print('epoch:{},ARI:{}'.format(epoch, mean(ARI)))

