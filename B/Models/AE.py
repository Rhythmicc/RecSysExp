from collections import OrderedDict

import torch.nn.functional as func
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

from B import *


class AutoEncoderDataset(data.Dataset):
    def __init__(self, dataset, n_user, n_item):
        self.n_user = n_user
        self.n_item = n_item
        self._x_mat = np.ones((n_item + 1, n_user + 1)) * 0
        self._mask = np.zeros((n_item + 1, n_user + 1))
        for u, v, r in dataset:
            self._x_mat[v][u] = r
            self._mask[v][u] = 1
        self.x_mat = torch.from_numpy(self._x_mat).float()
        self.mask = torch.from_numpy(self._mask).float()

    def __getitem__(self, index):
        return self.x_mat[index], self.mask[index]

    def __len__(self):
        return self.n_item

    def get_mat(self):
        return self.x_mat, self.mask

    def add_extra(self, dataset):
        if not dataset:
            return
        for u, v, r in dataset:
            self._x_mat[v][u] = r
            self._mask[v][u] = 1
        self.x_mat = torch.from_numpy(self._x_mat).float()
        self.mask = torch.from_numpy(self._mask).float()


class _AutoEncoder(torch.nn.Module):
    def __init__(self, n_user_and_rank, dropout=0.1):
        super(_AutoEncoder, self).__init__()
        encoder_dict = OrderedDict()
        for i in range(len(n_user_and_rank) - 1):
            encoder_dict['enc_linear' + str(i)] = torch.nn.Linear(n_user_and_rank[i], n_user_and_rank[i + 1])
            encoder_dict['enc_drop' + str(i)] = torch.nn.Dropout(dropout)
            encoder_dict['enc_relu' + str(i)] = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(encoder_dict)
        decoder_dict = OrderedDict()
        for i in range(len(n_user_and_rank) - 1, 0, -1):
            decoder_dict['dec_linear' + str(i)] = torch.nn.Linear(n_user_and_rank[i], n_user_and_rank[i - 1])
            decoder_dict['dec_drop' + str(i)] = torch.nn.Dropout(dropout)
            decoder_dict['dec_relu' + str(i)] = torch.nn.Sigmoid()
        self.decoder = torch.nn.Sequential(decoder_dict)

    def forward(self, x):
        x = (x - 1) / 5.0
        x = self.decoder(self.encoder(x))
        x = torch.clamp(x, 0, 1.0)  # torch.clamp(input, min, max)
        x = x * 5.0 + 1
        return x


class AutoEncoder:
    def __init__(self, n_user, n_item, rank, batch_size):
        self.train_set = None
        self.batch_size = batch_size
        self.net = _AutoEncoder([n_user + 1, rank])
        self.n_user = n_user
        self.n_item = n_item
        self.has_eval = False
        self.jump_store = 2
        self.store_step = 5
        self.part_set = None

    def forward(self, x):
        """使用x更新模型并返回结果和loss"""
        if not self.has_eval:
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            features = Variable(torch.FloatTensor(self.batch_size, self.n_user + 1))
            masks = Variable(torch.FloatTensor(self.batch_size, self.n_user + 1))

            for bid, (feature, mask) in enumerate(train_loader):
                if mask.shape[0] == self.batch_size:
                    features.data.copy_(feature)
                    masks.data.copy_(mask)
                else:
                    features = Variable(feature)
                    masks = Variable(mask)
                output = self.net(features)
                loss = func.mse_loss(output * masks, features * masks)
                loss.backward()
            return -1, 0
        else:
            res = []
            x_mat, mask = self.train_set.get_mat()
            features = Variable(x_mat)
            x_mat = self.net(features).t().cpu().data.numpy()
            rate_ls = [np.random.rand(), np.random.rand()]
            l, r = int(len(x) * min(rate_ls)), int(len(x) * max(rate_ls))
            if not self.part_set:
                self.part_set = x[l:r:self.store_step]
            else:
                self.part_set += x[l:r:self.store_step]  # 记录有价值的信息给下一轮训练
            for i, j, r in x:
                res.append(x_mat[i][j])
            return np.array(res), 0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        return self.net.train()

    def eval(self):
        self.has_eval = True
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def pre_deal_data(self, pd_data, start, end):
        u_ids = pd_data['user_id'][start: end].values  # [batch_size]
        i_ids = pd_data['item_id'][start: end].values  # [batch_size]
        labels = pd_data['label'][start: end].values

        res = []
        for index in range(len(u_ids)):
            res.append((u_ids[index], i_ids[index], labels[index]))
        return res

    def prepare_batches(self, pd_data):
        # * 产生data对应的所有batch的list，每个batch是一个dict，会被送入forward函数中
        data_list = self.pre_deal_data(pd_data, 0, len(pd_data))
        if not self.train_set or self.jump_store:
            self.train_set = AutoEncoderDataset(data_list, self.n_user, self.n_item)
            self.jump_store -= 1
        else:
            self.train_set.add_extra(data_list)
        self.train_set.add_extra(self.part_set)        # 载入传递训练数据
        self.part_set.clear() if self.part_set else 1  # 丢掉存储的训练数据
        self.has_eval = False
        total_batch = int((len(pd_data) + self.batch_size - 1) / self.batch_size)
        batches = list()
        for batch in range(total_batch):
            batch_start = batch * self.batch_size
            batch_end = min(len(pd_data), batch_start + self.batch_size)
            data_tuple = data_list[batch_start: batch_end]
            batches.append(data_tuple)
        return batches


def model(train_data, test_data):
    all_data = pd.concat([train_data, test_data])
    n_users = all_data['user_id'].unique().size
    n_items = all_data['item_id'].unique().size
    ae = AutoEncoder(n_users, n_items, 300, BATCH)
    return torch_runner(ae, train_data, test_data)
