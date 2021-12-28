import pandas as pd
from n_model import CloModel
import os
import pandas as pd
import torch
from tqdm import tqdm
from n_model import HCL
import dgl
import torch
import time
from sklearn.metrics import precision_recall_fscore_support
import argparse
from n_parse_code import clo_datasetSplit, DataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code Clone Detection")
    parser.add_argument('--pair_path', default='data/clone_pair/oj_clone_ids.pkl',
                        help='The path of the clone pairs')
    parser.add_argument('--data_path', default='data/oj/programs.pkl',
                        help='data path')
    parser.add_argument('--pre_model', default='model.pkl',
                        help='pre_model')
    parser.add_argument('--device', default='cpu',
                        help='device')
    args = parser.parse_args()

    '''
    变量进行保存
    '''
    pair_path = args.pair_path
    data_path = args.data_path

    device = args.device
    '''
        根据数据集需要更改的信息
        '''
    dataset_name = 'oj'

    train_ratio, val_ratio, test_ratio = 8, 1, 1
    train_data1, train_data2, train_data_label, valid_data1, valid_data2, valid_data_label, test_data1, test_data2, test_data_label \
        = clo_datasetSplit(pair_path,data_path,train_ratio,val_ratio,test_ratio)

    begin = 0

    lr, BATCH_SIZE, EPOCH = 0.00001, 2, 5
    USE_GPU = False
    in_feats, n_layer, n_head, drop_out, n_class = 768, 4, 4, 0.5, 1

    # pre_model = torch.load(args.pre_model)
    # pre_model_dict = pre_model.state_dict()
    model = CloModel(in_feats, n_layer, drop_out, n_class, device='cpu').to(device)
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.BCELoss()
    precision, recall, f1 = 0, 0, 0

    pbar = tqdm(range(EPOCH))

    best_model = model
    best_acc = 0.0
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    dataSet = DataSet()

    for t in range(1, n_class + 1):
        train_data1_t, train_data2_t, test_data1_t, test_data2_t, train_label_t, test_label_t = [], [],  [], [], [], []
        if dataset_name == 'bcb':
            for i in range(len(train_data_label)):
                if train_data_label == t:
                    train_data1_t.append(train_data1[i])
                    train_data2_t.append(train_data2[i])
                    train_label_t.append[1]

            for i in range(len(test_data_label)):
                if test_data_label[i] == t:
                    test_data1_t.append(test_data1[i])
                    test_data2_t.append(test_data2[i])
                    test_label_t.append[1]

        else:
            train_data1_t, train_data2_t, test_data1_t, test_data2_t = train_data1, train_data2, test_data1, test_data2
            train_label_t,test_label_t = train_data_label, test_data_label

        for epoch in pbar:
            pbar.set_description('epoch:%d  processing' % (epoch))
            i = 0
            start_time = time.time()
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            model.train()

            while (i + BATCH_SIZE) <= len(train_data1_t):
                model.train()

                input_node1, path1, label1, p_n_dict1, n_ast_node1, src1, dst1, downtown_label1 = dataSet.parse_c(
                    train_data1.iloc[i:i + BATCH_SIZE, :], dataset_name)
                input_node2, path2, label2, p_n_dict2, n_ast_node2, src2, dst2, downtown_label2 = dataSet.parse_c(
                    train_data2.iloc[i:i + BATCH_SIZE, :], dataset_name)
                label_batch = train_label_t[i:i+BATCH_SIZE]


                i = i + BATCH_SIZE

                '''
                输入数据类型转换，并选择需要的运行环境
                '''
                g1 = dgl.graph((src1, dst1))
                g1 = dgl.add_self_loop(g1).to(device)
                g2 = dgl.graph((src2, dst2))
                g2 = dgl.add_self_loop(g2).to(device)

                node_emb1 = torch.FloatTensor(input_node1).to(device)
                path_emb1 = torch.FloatTensor(path1).to(device)
                node_emb2 = torch.FloatTensor(input_node2).to(device)
                path_emb2 = torch.FloatTensor(path2).to(device)
                labels = torch.FloatTensor(label_batch).to(device)



                model.zero_grad()
                logits = model(g1, node_emb1, n_ast_node1, path_emb1, p_n_dict1, g2, node_emb2, n_ast_node2, path_emb2, p_n_dict2)

                labels = labels.view(-1, 1)
                print('n_clo',labels.shape)
                loss = loss_function(logits, labels)

                loss.backward()
                optimizer.step()

        print("Testing-%d...")
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0

        while (i + BATCH_SIZE) < len(test_data1_t):
            input_node1, path1, label1, p_n_dict1, n_ast_node1, src1, dst1, downtown_label1 = dataSet.parse_c(
                test_data1.iloc[i:i + BATCH_SIZE, :], dataset_name)
            input_node2, path2, label2, p_n_dict2, n_ast_node2, src2, dst2, downtown_label2 = dataSet.parse_c(
                test_data2.iloc[i:i + BATCH_SIZE, :], dataset_name)
            label_batch = test_label_t[i:i + BATCH_SIZE]


            i = i + BATCH_SIZE
            g1 = dgl.graph((src1, dst1))
            g1 = dgl.add_self_loop(g1).to(device)
            g2 = dgl.graph((src2, dst2))
            g2 = dgl.add_self_loop(g2).to(device)

            node_emb1 = torch.FloatTensor(input_node1).to(device)
            path_emb1 = torch.FloatTensor(path1).to(device)
            node_emb2 = torch.FloatTensor(input_node2).to(device)
            path_emb2 = torch.FloatTensor(path2).to(device)
            labels = torch.FloatTensor(label_batch).to(device)

            model.zero_grad()
            logits = model(g1, node_emb1, n_ast_node1, path_emb1, p_n_dict1, g2, node_emb2, n_ast_node2, path_emb2, p_n_dict2)

            labels = labels.view(-1, 1)
            loss = loss_function(logits, labels)

            # calc testing acc
            predicted = (logits.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(labels.cpu().numpy())
            total += len(labels)
            total_loss += loss.item() * len(labels)
        if dataset_name == 'bcb':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
