from B import *


class MF(torch.nn.Module):
    def __init__(self, n_users, n_items):
        super(MF, self).__init__()
        self.user_num = n_users
        self.item_num = n_items
        self._define_params()

    def _define_params(self):
        self.u_embeddings = torch.nn.Embedding(self.user_num + 1, EMB_SIZE)
        self.i_embeddings = torch.nn.Embedding(self.item_num + 1, EMB_SIZE)

    @staticmethod
    def init_weights(m):
        if 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, feed_dict):
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size]
        labels = feed_dict['label']

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # 内积
        loss = ((labels - prediction) ** 2).mean().sqrt()  # RMSE作为loss
        return prediction.data.numpy(), loss
    
    def prepare_batches(self, data):
        # 产生data对应的所有batch的list，每个batch是一个dict，会被送入forward函数中
        total_batch = int((len(data) + BATCH - 1) / BATCH)
        batches = list()
        for batch in range(total_batch):
            batch_start = batch * BATCH
            batch_end = min(len(data), batch_start + BATCH)
            user_ids = data['user_id'][batch_start: batch_end].values
            item_ids = data['item_id'][batch_start: batch_end].values
            labels = data['label'][batch_start: batch_end].values
            feed_dict = {
                'user_id': torch.from_numpy(user_ids),
                'item_id': torch.from_numpy(item_ids),
                'label': torch.from_numpy(labels).float()
            }
            batches.append(feed_dict)
        return batches


def model(train_data, test_data):
    all_data = pd.concat([train_data, test_data])
    n_users = all_data['user_id'].unique().size
    n_items = all_data['item_id'].unique().size
    mf = MF(n_users, n_items)  # 定义模型对象
    mf.apply(mf.init_weights)
    return torch_runner(mf, train_data, test_data)
