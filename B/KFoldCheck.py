from . import *
from sklearn.model_selection import KFold


def cross_validation(data_df, fit_predict):
    kf = KFold(n_splits=5)  # 调用sklearn相关函数进行数据集划分
    kf.get_n_splits(data_df)
    rmse_lst = list()
    for k, (train_index, test_index) in enumerate(kf.split(data_df)):  # 5折交叉验证
        print('Fold', k)
        train_data = data_df.iloc[train_index]
        test_data = data_df.iloc[test_index]
        prediction = fit_predict(train_data, test_data)  # 获得预测评分
        rmse = np.sqrt(mean_squared_error(test_data['label'], prediction))  # 计算rmse
        rmse_lst.append(rmse)
        print('RMSE: {:<.4f}\n'.format(rmse))
    print('Average: {:<.4f}'.format(np.mean(rmse_lst)))
