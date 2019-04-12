import lightgbm as lgb
import pandas as pd
import numpy as np
import multiprocessing
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier, Pool

path = Path("data")
# train = pd.read_csv(path / "train.csv")
# test = pd.read_csv(path / "test.csv")


# 多线程读取
def load_dataframe(dataset):
    return pd.read_csv(dataset)


with multiprocessing.Pool() as pool:
    train_df, valid_df, test_df = pool.map(
        load_dataframe, [path / 'train.csv', path / 'valid.csv', path / 'test.csv'])
print("loaded data!")


# train = train_df.drop(columns=["ID_code"])
# test = test_df.drop(columns=["ID_code"])


# # csv 文件切片存储验证集
# valid = train_df.loc[160000:200001]
# filename = path / "valid.csv".format(datetime.now())
# valid.to_csv(filename, index=False)
# train_df = train_df.loc[0:160000]


# Inspiration from
# https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment

# 试试转换
# for i in range(0,200):
#     train["var_"+str(i)] = train["var_"+str(i)]*(np.log(np.abs(2*train["var_"+str(i)])))
#     test["var_"+str(i)] = test["var_"+str(i)]*(np.log(np.abs(2*test["var_"+str(i)])))
#     print(train.head(5))
# train.to_csv("train_trans.csv", index=False)
# test.to_csv("test_trans.csv",index = False)


# 手动增加特征
# columns = []
# for i in range(0,200):
#     columns.append("var_"+str(i))
# print(columns)

# 'var_108', 'var_126',
def transform(df, num, var):
    df['random_{}'.format(var)] = np.random.normal(
        df[var].mean(), df[var].std(), num).round(4)
    var_counts = pd.DataFrame(df.groupby(var)['ID_code'].count()).reset_index()
    var_counts_random = pd.DataFrame(df.groupby('random_{}'.format(var))[
                                     'ID_code'].count()).reset_index()
    merged_counts = pd.merge(
        var_counts, var_counts_random, left_on=var, right_on='random_{}'.format(var))
    merged_counts['diff'] = merged_counts[
        'ID_code_x'] - merged_counts['ID_code_y']
    df['{}_diff_normal_dist'.format(var)] = df.merge(
        merged_counts[[var, 'diff']], how='left')['diff']
    df = df.drop('random_{}'.format(var), axis=1)
    #df = df.drop(var, axis=1)
    return df


# train_df = train_df.drop(columns=["ID_code"])
# print("train")
# # valid = valid_df.drop(columns=["ID_code"])
# # print(valid)
# test_df = test_df.drop(columns=["ID_code"])
# print("test")


# # columns = ['var_0', 'var_1', 'var_2', 'var_3', 'var_4','var_12', 'var_13', 'var_108', 'var_126', 'var_68']
# for var in train_df.columns:
#     print(var)
#     hist, bin_edges = np.histogram(train_df[var], bins=1000, density=True)
#     train_df['test_' +
# var] = [hist[np.searchsorted(bin_edges, ele) - 1] for ele in
# train_df[var]]


# for var in test_df.columns:
#     print(var)
#     hist, bin_edges = np.histogram(test_df[var], bins=1000, density=True)
#     test_df['test_' +
# var] = [hist[np.searchsorted(bin_edges, ele) - 1] for ele in
# test_df[var]]


# 3 4 5 7 8 10
for var in tqdm(['var_{}'.format(x) for x in range(0, 200)]):
    print(var)
    train_df = transform(train_df, var=var, num=200000)
    # print(train_df)
    test_df = transform(test_df, var=var, num=200000)

# Loop and add features 2019_04_02_19_43
# for var in tqdm(['var_{}'.format(x) for x in range(0, 200)]):
#     pd.concat([train_df, transform(train_df, var=var, num=200000)])
#     # print(train_df)
#     pd.concat([test_df, transform(test_df, var=var, num=200000)])

# train_df.fillna(0)

train_df.where(train_df.isnull(), 0.0)
train = train_df.drop(columns=["ID_code"])
print(train)
# valid = valid_df.drop(columns=["ID_code"])
# print(valid)
test_df.where(test_df.isnull(), 0.0)
test = test_df.drop(columns=["ID_code"])
print(test)
# train = train_df.drop(columns=["ID_code"])
# test = test_df.drop(columns=["ID_code"])


def augment(train, num_n=1, num_p=2):
    newtrain = [train]

    n = train[train.target == 0]
    for i in range(num_n):
        newtrain.append(
            n.apply(lambda x: x.values.take(np.random.permutation(len(n)))))

    for i in range(num_p):
        p = train[train.target > 0]
        newtrain.append(
            p.apply(lambda x: x.values.take(np.random.permutation(len(p)))))
    return pd.concat(newtrain)
# # df=oversample(train,2,1)


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.055,
    'learning_rate': 0.0055,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 70,
    'min_sum_hessian_in_leaf': 9.0,
    'num_leaves': 20,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
}


result = np.zeros(test.shape[0])


rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=2, random_state=10)
for counter, (train_index, valid_index) in enumerate(rskf.split(train, train.target), 1):
    print(counter)

    # Train data
    t = train.iloc[train_index]
    t = augment(t)
    print(t)
    trn_data = lgb.Dataset(t.drop("target", axis=1), label=t.target)

    # Validation data
    v = train.iloc[valid_index]
    val_data = lgb.Dataset(v.drop("target", axis=1), label=v.target)

    # Training
    model = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data,
                                                            val_data], verbose_eval=500, early_stopping_rounds=3000)
    result += model.predict(test)


# 自己分割数据集
# for counter in range(1, 4):
#     print("当前:", counter)
#     # Train data
#     t = train
#     # print(t.head())
#     # t = augment(t)
#     trn_data = lgb.Dataset(t.drop("target", axis=1), label=t.target)
#     # Val data
#     v = valid
#     val_data = lgb.Dataset(v.drop("target", axis=1), label=v.target)
#     # Training
#     model = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data,
# val_data], verbose_eval=500, early_stopping_rounds=3000)

#     result += model.predict(test)


submission = pd.read_csv(path / 'sample_submission.csv')
submission['target'] = result / counter
filename = "{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())
submission.to_csv(filename, index=False)


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X_test = test_df.drop('ID_code', axis=1)
X = train_df.drop(['ID_code', 'target'], axis=1)
y = train_df['target']

train_pool = Pool(X, y)
m = CatBoostClassifier(iterations=300, eval_metric="AUC",
                       boosting_type='Ordered')
m.fit(X, y, silent=True)
y_pred1 = m.predict(X_test)

sub1 = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub1["target"] = y_pred1
sub1.to_csv("submission1.csv", index=False)
