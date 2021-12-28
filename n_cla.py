from tqdm import tqdm
from n_model import ClaModel
import dgl
import torch
import torch.nn.functional as F
import time
from n_parse_code import datasetSplit,DataSet

def CrossEntropyLoss_label_smooth(outputs, targets,device, num_classes=104, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1)).to(device)
    targets = targets.data
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    log_prob = F.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


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

    '''
    根据数据集需要更改的信息
    '''
    dataset_name = 'oj'
    n_class = 104




    # pre_model_dict = torch.load(args.pre_model)

    train_ratio, val_ratio, test_ratio = 8, 1, 1
    train_data,val_data,test_data = datasetSplit(data_path,train_ratio,val_ratio,test_ratio,data_suf='pkl')
    dataSet = DataSet()

    begin = 0
    '''
    定义一些超参数
    '''
    lr, BATCH_SIZE, EPOCH = 0.00001,2, 150
    in_feats,n_layer,drop_out=768,2,0.5
    n_path_node = 13




    '''
    对模型进行定义
    '''
    model = ClaModel(in_feats,n_path_node,n_layer,drop_out,n_class, device = device).to(device)
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = CrossEntropyLoss_label_smooth
    pbar = tqdm(range(EPOCH))

    best_model = model
    best_acc = 0.0
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []

    for epoch in pbar:
        pbar.set_description('epoch:%d  processing' % (epoch))
        i = 0
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        model.train()

        while (i + BATCH_SIZE) <= len(train_data):
            model.train()

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
            downtown_label = torch.tensor(downtown_label).to(device)

            model.zero_grad()
            logits = model(g,input_node, n_ast_node,path, p_n_dict)
            loss = loss_function(logits, downtown_label, device)

            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(logits.data, 1)
            total_acc += (predicted == downtown_label).sum()

            total += len(downtown_label)
            total_loss += loss.item() * len(downtown_label)
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)

        i = 0
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        while (i + BATCH_SIZE) <= len(val_data):

            model.eval()
            input_node, path, label, p_n_dict, n_ast_node, src, dst, downtown_label = dataSet.parse_c(val_data.iloc[i:i + BATCH_SIZE, :], dataset_name)
            i = i + BATCH_SIZE

            g = dgl.graph((src, dst))
            g = dgl.add_self_loop(g)
            input_node = torch.FloatTensor(input_node).to(device)
            path = torch.FloatTensor(path).to(device)
            downtown_label = torch.tensor(downtown_label).to(device)
            logits = model(g,input_node, n_ast_node,path, p_n_dict)

            _, predicted = torch.max(logits.data, 1)
            total_acc += (predicted == downtown_label).sum()

            total += len(downtown_label)
            total_loss += loss.item() * len(downtown_label)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc / total > best_acc:
            best_model = model
            torch.save(best_model, 'bestmodel.pkl')
            best_acc=total_acc / total
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCH, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while (i + BATCH_SIZE) <= len(test_data):

        model.eval()
        input_node, path, label, p_n_dict, n_ast_node, src, dst, downtown_label = dataSet.parse_c(test_data.iloc[i:i + BATCH_SIZE, :], dataset_name)
        i = i + BATCH_SIZE
        g = dgl.graph((src, dst))
        g = dgl.add_self_loop(g)

        input_node = torch.FloatTensor(input_node).to(device)
        path = torch.FloatTensor(path).to(device)
        downtown_label = torch.tensor(downtown_label).to(device)
        logits = model(g,input_node, n_ast_node,path, p_n_dict)

        # calc training acc
        _, predicted = torch.max(logits.data, 1)
        total_acc += (predicted == downtown_label).sum()
        total += len(downtown_label)
        total_loss += loss.item() * len(downtown_label)
    print("Testing results(Acc):", total_acc / total)
