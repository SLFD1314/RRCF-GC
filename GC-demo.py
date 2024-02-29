import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from model.GC import Net
from DataProcessing import DataProcessing
from ParaInitialization.ParaInitialization import initialize_arguments

args = initialize_arguments()

dataset = np.load(args.dataset)
train_raw_data = dataset[args.traindata]
test_raw_data = dataset[args.testdata]

task = {'arr_0': 'task5'}
parent_folder  = 'result'
file_task_names = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
for file_task_name in file_task_names:
    if task[args.traindata] == file_task_name:
        result_folder = os.path.join(parent_folder, file_task_name)
        file_parameter_names = [f for f in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, f))]
        for file_parameter_name in file_parameter_names:
            split_len_names = file_parameter_name.split('_')
            if args.seq_len == int(split_len_names[2]) and args.pred_len == int(split_len_names[5]):
                parameter_folder = os.path.join(result_folder, file_parameter_name)
                data_train_path = os.path.join(parameter_folder, 'TrainsetOutput')
                train_output_data = DataProcessing.read_all_csv(data_train_path)
                data_test_path = os.path.join(parameter_folder, 'TestsetOutput')
                test_output_data = DataProcessing.read_all_csv(data_test_path)

train_raw_data[:, -args.pred_len:, :] = train_output_data
test_raw_data[:, -args.pred_len:, :] = test_output_data
traindata = train_raw_data
testdata = test_raw_data

labels_dom_train = np.zeros([traindata.shape[0], 1]).astype(int)
labels_dom_test = np.ones([testdata.shape[0], 1]).astype(int)

labels_cla_train = dataset[args.trainlabel]
labels_cla_train = labels_cla_train.reshape(-1, 1)
labels_cla_test = dataset[args.testlabel]
labels_cla_test = labels_cla_test.reshape(-1, 1)

train_data = torch.FloatTensor(traindata).transpose(2,1)
test_data = torch.FloatTensor(testdata).transpose(2,1)
train_label_dom = torch.LongTensor(labels_dom_train)
test_label_dom = torch.LongTensor(labels_dom_test)
train_label_cla = torch.LongTensor(labels_cla_train)
test_label_cla = torch.LongTensor(labels_cla_test)

train_set = TensorDataset(train_data, train_label_cla, train_label_dom)
test_set = TensorDataset(test_data, test_label_cla, test_label_dom)

src_train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
tar_train_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
src_test_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
tar_test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
if args.use_gpu:
    loss_fn = loss_fn.cuda()

epoch = 10000
lr = 0.001
def train(net):
    total_train_step = 0
    src_iter = iter(src_train_dataloader)
    tar_iter = iter(tar_train_dataloader)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for i in range(1, 1+epoch):
        net.train()
        try:
            src_data, src_label_cla, src_label_dom = src_iter.next()
        except Exception as err:
            src_iter = iter(src_train_dataloader)
            src_data, src_label_cla, src_label_dom = src_iter.next()
        try:
            tar_data, tar_label_cla, tar_label_dom = tar_iter.next()
        except Exception as err:
            tar_iter = iter(tar_train_dataloader)
            tar_data, tar_label_cla, tar_label_dom = tar_iter.next()

        if torch.cuda.is_available():
            src_data = src_data.cuda()
            src_label_cla = src_label_cla.cuda()
            src_label_dom = src_label_dom.cuda()
            tar_data = tar_data.cuda()
            tar_label_dom = tar_label_dom.cuda()

        alpha = 2. / (1. + np.exp(-10 * (i) / epoch)) - 1
        src_class_output, src_domain_output, tar_domain_output = net(src_data, tar_data, alpha=alpha)
        loss_cla = F.nll_loss(F.log_softmax(src_class_output, dim=1), src_label_cla.squeeze(1).long())
        loss_dom_s= F.nll_loss(F.log_softmax(src_domain_output, dim=1), src_label_dom.squeeze(1).long())
        loss_dom_t = F.nll_loss(F.log_softmax(tar_domain_output, dim=1), tar_label_dom.squeeze(1).long())

        loss = loss_cla  + alpha*(loss_dom_s + loss_dom_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            Source_accuracy, Source_loss = test(net, src_test_dataloader)
            Target_accuracy, Target_loss = test(net, tar_test_dataloader)

            print("Iter: {0}  | Train Acc: {1:.2f}   Test Acc: {2:.2f}   Train Loss: {3:.7f}   Test Loss: {4:.7f}".format(
                total_train_step, Source_accuracy, Target_accuracy, Source_loss, Target_loss))


def test(net,test_dataloader):
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data, label_cla, label_dom in test_dataloader:
            if torch.cuda.is_available():
                data = data.cuda()
                label_cla = label_cla.cuda()
            outputs_cla = net.predict(data)
            loss_cla = loss_fn(outputs_cla, label_cla.squeeze(1).long())
            total_test_loss = total_test_loss + len(label_cla)*loss_cla.item()
            accuracy = (outputs_cla.argmax(1) == label_cla.squeeze(1).long()).sum()
            total_accuracy = total_accuracy + accuracy
        acc = 100. * total_accuracy / len(test_dataloader.dataset)
        loss = total_test_loss /  len(test_dataloader.dataset)
    return acc, loss


if __name__ == '__main__':
    net = Net()
    if torch.cuda.is_available():
        net = net.cuda()
    train(net)