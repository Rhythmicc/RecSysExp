from B import pd


def import_movieLens_100k_data():
    df = pd.read_csv('./data/ml-100k/u.data', sep='\t', header=None)
    df.columns = ['user_id', 'item_id', 'label', 'time']
    return df


def import_movieLens_100k_item():
    df = pd.read_csv('./data/ml-100k/u.item', sep='|', header=None, encoding="ISO-8859-1")
    df = df.drop([1, 2, 3, 4], axis=1)
    df.columns = [
        'item_id', 'Action', 'Adventure', 'Animation', "Children's",
        'Comedy', 'Crime', 'Documentary ', 'Drama ', 'Fantasy ',
        'Film-Noir ', 'Horror ', 'Musical ', 'Mystery ', 'Romance ',
        'Sci-Fi ', 'Thriller ', 'War ', 'Western', 'Other'
    ]
    return df


def import_movieLens_100k_user():
    df = pd.read_csv('./data/ml-100k/u.user', sep='|', header=None)
    df = df[[0, 1, 2, 3]]
    df.columns = ['user_id', 'Age', 'Gender', 'Occupation']
    return df


def import_amazon_data():
    import gzip

    def parse(path):
        for l in gzip.open(path, 'rb'):
            yield eval(l)

    def get_df(path):
        i, df = 0, {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    amazon_data_df = get_df('./data/amazon/reviews_Cell_Phones_and_Accessories_5.json.gz')
    amazon_data_df.rename(columns={'asin': 'item_id',
                                   'reviewerID': 'user_id',
                                   'unixReviewTime': 'time',
                                   'overall': 'label'}, inplace=True)
    amazon_data_df = amazon_data_df[['user_id', 'item_id', 'label', 'time']]

    # 重编号
    uids = sorted(amazon_data_df['user_id'].unique())
    user2id = dict(zip(uids, range(1, len(uids) + 1)))
    iids = sorted(amazon_data_df['item_id'].unique())
    item2id = dict(zip(iids, range(1, len(iids) + 1)))
    amazon_data_df['user_id'] = amazon_data_df['user_id'].apply(lambda x: user2id[x])
    amazon_data_df['item_id'] = amazon_data_df['item_id'].apply(lambda x: item2id[x])
    # 随机打乱
    amazon_data_df = amazon_data_df.sample(frac=1, random_state=0)
    return amazon_data_df
