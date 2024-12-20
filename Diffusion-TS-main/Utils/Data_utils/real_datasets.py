import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        in_len=96,
        out_len=24, 
        data_split = [8640, 2880, 2880],
        save2npy=True, 
        seed=123,
        period='train',
        output_dir='./OUTPUT',
    ):
        super(CustomDataset, self).__init__()
        print('data root:', data_root)
        self.name = name
        self.in_len, self.out_len = in_len, out_len
        self.data_split = data_split
        self.save2npy = save2npy
        self.seed = seed
        self.period = period
        self.output_dir = output_dir
        
        assert period in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[period]
        self.window = in_len + out_len

        #read data and normalize with mean and std of training data
        self.normed_data = self.read_data(data_root, self.name)
        self.var_num = self.normed_data.shape[-1]
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        train, val, inference = self.__getsamples(self.normed_data, self.data_split, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            masks = np.ones(self.samples.shape)
            masks[:, -self.out_len:, :] = 0
            self.masking = masks.astype(bool)
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(data)*self.data_split[0]); 
            test_num = int(len(data)*self.data_split[2])
            val_num = len(data) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
        
        #train samples
        train_sample_num = border2s[0] - border1s[0] - self.window + 1
        train_x = np.zeros((train_sample_num, self.window, self.var_num))
        for i in range(train_sample_num):
            start = i
            end = i + self.window
            train_x[i, :, :] = data[start:end, :]
        
        #val samples
        val_sample_num = border2s[1] - border1s[1] - self.window + 1
        val_x = np.zeros((val_sample_num, self.window, self.var_num))
        for i in range(val_sample_num):
            start = border1s[1] + i
            end = border1s[1] + i + self.window
            val_x[i, :, :] = data[start:end, :]
        
        #test samples
        test_sample_num = border2s[2] - border1s[2] - self.window + 1
        test_x = np.zeros((test_sample_num, self.window, self.var_num))
        for i in range(test_sample_num):
            start = border1s[2] + i
            end = border1s[2] + i + self.window
            test_x[i, :, :] = data[start:end, :]

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_x)
            np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_val.npy"), val_x)
            np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_x)

        return train_x, val_x, test_x

    def read_data(self, filepath, name=''):
        """Reads a single .csv
        """
        print(f"Reading {name} data from {filepath}")

        df_raw = pd.read_csv(filepath, header=0)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        self.scaler = StandardScaler()
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)

        whole_data = self.scaler.transform(df_data.values)

        return whole_data

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
