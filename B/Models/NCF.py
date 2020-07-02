from B import *


class NCF(torch.nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        """
        self.batch_size = BATCH
        self.embed_user_GMF = torch.nn.Embedding(user_num + 1, factor_num)
        self.embed_item_GMF = torch.nn.Embedding(item_num + 1, factor_num)
        self.embed_user_MLP = torch.nn.Embedding(user_num + 1, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = torch.nn.Embedding(item_num + 1, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(torch.nn.Dropout(p=dropout))
            MLP_modules.append(torch.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(torch.nn.ReLU())
        self.MLP_layers = torch.nn.Sequential(*MLP_modules)

        self.predict_layer = torch.nn.Linear(factor_num * 2, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        torch.nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        torch.nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, feed_dict):
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size]
        labels = feed_dict['label']

        embed_user_GMF = self.embed_user_GMF(u_ids)
        embed_item_GMF = self.embed_item_GMF(i_ids)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(u_ids)
        embed_item_MLP = self.embed_item_MLP(i_ids)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        loss = ((labels - prediction) ** 2).mean().sqrt()  # RMSE作为loss
        return prediction, loss

    def prepare_batches(self, data):
        # 产生data对应的所有batch的list，每个batch是一个dict，会被送入forward函数中
        total_batch = int((len(data) + self.batch_size - 1) / self.batch_size)
        batches = list()
        for batch in range(total_batch):
            batch_start = batch * self.batch_size
            batch_end = min(len(data), batch_start + self.batch_size)
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
    ncf = NCF(n_users, n_items, EMB_SIZE, 1, 0.2)  # 定义模型对象
    return torch_runner(ncf, train_data, test_data)
