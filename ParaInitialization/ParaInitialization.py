import argparse
import random
import torch
import numpy as np


def initialize_arguments():
    parser = argparse.ArgumentParser(description='RRCF-GC')

    # random seed
    parser.add_argument('--random_seed', type=int, default=7, help='random seed')

    # data loader
    parser.add_argument('--dataset', type=str, default='data.npz', help='save normalization raw_data')
    parser.add_argument('--num_sensors', type=int, default=8, help='number of sensors')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=200, help='prediction sequence length')
    parser.add_argument('--traindata', type=str, default='arr_0', help='trainset')
    parser.add_argument('--testdata', type=str, default='arr_1', help='testset')
    parser.add_argument('--trainlabel', type=str, default='arr_2', help='trainset')
    parser.add_argument('--testlabel', type=str, default='arr_3', help='trainset')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

    # optimization
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)
    print("********************************************************************")
    return args