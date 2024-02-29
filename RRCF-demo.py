from DataProcessing import DataProcessing
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from model import RRCF
import torch.nn as nn
from torch import optim
import time
from ParaInitialization.ParaInitialization import initialize_arguments

args = initialize_arguments()

def vali(model, vali_loader, criterion):
    total_loss = []
    model.eval()
    out = []
    with torch.no_grad():
        for i, data in enumerate(vali_loader):
            if args.use_gpu:
                data = data.cuda()
            outputs = model(data[:, :args.seq_len, :])
            label = data[:, -args.pred_len:, :]
            pred = outputs.detach().cpu()
            true = label.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)
            out.append(outputs)
    output_test = torch.cat(out, dim=0)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss, output_test


if __name__ == '__main__':

    dataset = np.load(args.dataset)
    trainset = dataset[args.traindata]
    testset = dataset[args.testdata]

    train_data = torch.FloatTensor(trainset)
    test_data = torch.FloatTensor(testset)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    src_test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    tar_test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    train_steps = len(train_loader)
    model = RRCF.Model(args).float()
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    if args.use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    opt_test = 100
    k = 0
    for epoch in range(args.train_epochs):
        train_loss = []
        model.train()
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            if args.use_gpu:
                data = data.cuda()
            model_optim.zero_grad()
            outputs = model(data[:, :args.seq_len, :])
            label = data[:, -args.pred_len:, :]
            loss = criterion(outputs, label)
            loss.backward()
            model_optim.step()
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

        train_loss, output_train = vali(model, src_test_loader, criterion)
        test_loss, output_test = vali(model, tar_test_loader, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Test Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, test_loss))


        if test_loss < opt_test:
            opt_output_train = output_train
            opt_output_test = output_test

        parent_folder  = 'result'
        os.makedirs(parent_folder, exist_ok=True)

        task = {'arr_0': 'task5'}
        file_task_name = str(task[args.traindata])
        result_folder = os.path.join(parent_folder, file_task_name)
        os.makedirs(result_folder, exist_ok=True)

        file_parameter_name = 'seq_len_' + str(args.seq_len) + '_pred_len_' + str(args.pred_len)
        parameter_folder = os.path.join(result_folder, file_parameter_name)
        os.makedirs(parameter_folder, exist_ok=True)

    data_train_path = os.path.join(parameter_folder, 'TrainsetOutput')
    DataProcessing.save_outdata(data_train_path, opt_output_train)

    data_test_path = os.path.join(parameter_folder, 'TestsetOutput')
    DataProcessing.save_outdata(data_test_path, opt_output_test)


