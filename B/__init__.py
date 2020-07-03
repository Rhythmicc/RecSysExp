import gc

import numba
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

__name__ = "B"
__package__ = "B"
SEED = 2020  # 随机数种子，方便重现实验结果
LR = 1e-2  # 学习率
L2 = 1e-4  # 权重衰减系数
EPOCH = 100  # 训练的总轮次
BATCH = 1024  # 批大小
EMB_SIZE = 10  # MF模型的嵌入维度


def random_model(train_data, test_data):
    return np.random.randint(1, 6, len(test_data))


def torch_runner(model, train_data, test_data):
    # 设置随机数种子，定义优化器
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

    # 这里模型类内部需要实现一个准备batch的函数，将数据分批，并定义每批的输入是什么
    test_batches = model.prepare_batches(test_data)
    best_predictions, test_results = None, list()
    for epoch in range(EPOCH):  # 一共训练EPOCH轮
        gc.collect()
        epoch_train_data = train_data.copy().sample(frac=1)  # 随机打乱训练集
        batches = model.prepare_batches(epoch_train_data)  # 准备训练集batch

        # 一轮训练
        model.train()
        loss_lst = list()
        print_flag = True
        for batch in batches:
            optimizer.zero_grad()
            prediction, loss = model(batch)
            try:
                loss.backward()
                loss_lst.append(loss.detach().cpu().data.numpy())
            except:
                print_flag = False
            optimizer.step()

        # 测试结果
        model.eval()
        predictions = list()
        for batch in test_batches:
            prediction, loss = model(batch)
            predictions.extend(prediction)
        rmse = np.sqrt(mean_squared_error(test_data["label"], predictions))
        if epoch == 0 or rmse < min(test_results):  # 如果当前的模型是目前最好的
            best_predictions = predictions
        test_results.append(rmse)

        if print_flag:
            print("Epoch {:<3} loss={:<.4f}\t test={:<.4f}".format(epoch + 1, np.mean(loss_lst), rmse), end="\r")
        else:
            print("Epoch {:<3}\t test={:<.4f}".format(epoch + 1, rmse), end="\r")
    print()
    return best_predictions
