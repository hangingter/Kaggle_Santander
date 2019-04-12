import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool

# sub1 = pd.read_csv('Bayes.csv')
# sub2 = pd.read_csv('Gaussian.csv')
# sub3 = pd.read_csv('2019-03-28_21_14_sub.csv')
# sub4 = pd.read_csv("blended_submission_2019-04-01_10_31.csv")
# sub5 = pd.read_csv("2019-04-08_18_08_sub.csv")


# # sub1.target = sub1.target * 0.15 + sub2.target * 0.05 + sub3.target * 0.7
# sub1.target = sub1.target * 0.07 + sub2.target * 0.01 + sub3.target * 0.92
# sub1.target = sub1.target * 0.08 + sub3.target * 0.92
# 目前最好：
# sub1.target = sub4.target * 0.90 + sub3.target * 0.05 + sub5.target * 0.05


# test 2539
# for i in range(0,200000):
# 	print(sub1.loc[i,"target"])
# 	if sub1.loc[i,"target"] < 0.001:
# 		sub1.loc[i,"target"] = 0
# 	if sub1.loc[i,"target"] > 0.98:
# 		sub1.loc[i,"target"] =1

# sub1.target = sub1.target * 0.08 + sub2.target * 0.02 + sub4.target * 0.08 + sub3.target * 0.82

# sub1.to_csv('ensemble.csv', index=False)


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')



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


for var in tqdm(['var_{}'.format(x) for x in range(0, 200)]):
    print(var)
    train_df = transform(train_df, var=var, num=200000)
    # print(train_df)
    test_df = transform(test_df, var=var, num=200000)

train_df = train_df.drop(['ID_code'], axis=1)
train_df = augment(train_df)
X_test = test_df.drop('ID_code', axis=1)
X = train_df.drop(['target'], axis=1)
y = train_df['target']

train_pool = Pool(X, y)
m = CatBoostClassifier(iterations=300, eval_metric="AUC",
                       boosting_type='Ordered')
m.fit(X, y, silent=True)
y_pred1 = m.predict(X_test)

sub1 = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub1["target"] = y_pred1
sub1.to_csv("submission2.csv", index=False)
