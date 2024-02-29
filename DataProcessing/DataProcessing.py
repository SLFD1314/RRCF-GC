import numpy as np
import os
import warnings
from pandas import DataFrame
import shutil
import pandas as pd
warnings.filterwarnings('ignore')


def save_outdata(file, output_data):
    if os.path.exists(file + '\\' + 'OutputData' + str(3000) + '.csv'):
        shutil.rmtree(file)

    os.makedirs(file)
    for i in range(0, output_data.shape[0]):
        for j in range(0, output_data.shape[1]):
            list = output_data[i, j, :].tolist()
            data = DataFrame([list])
            dir = file + '\\' + 'OutputData' + str(i) + '.csv'
            data.to_csv(dir, mode='a', header=False, index=False)


def read_all_csv(folder_path):
    all_data = []
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        csv_files = [file for file in files if file.endswith('.csv')]
        sorted_files = sorted(csv_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))))
        for file_name in sorted_files:
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, header=None).values
            all_data.append(data)
        output_data = np.array(all_data)
    else:
        print(f'Folder {folder_path} does not exist.')

    return output_data






