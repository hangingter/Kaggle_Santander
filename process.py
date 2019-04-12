
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")

X_train = raw_train.drop(columns=["ID_code", "target"])
y_train = raw_train[["target"]]
X_test = raw_test.drop(columns=["ID_code"])
y_test = raw_test[["ID_code"]]


scaler = StandardScaler()

print(X_train)
# Check y training data distribution
sns.countplot(x="target", data=y_train)
print(y_train['target'].value_counts() / y_train.shape[0])
print('{} samples are positive'.format(np.sum(y_train['target'] == 1)))
print('{} samples are negative'.format(np.sum(y_train['target'] == 0)))

# 检查异常值
df1 = pd.concat([X_train.apply(lambda x: sum(x.isnull())).rename("num_missing"),
                 X_train.apply(lambda x: sum(x == 0)).rename("num_zero"),
                 X_train.apply(lambda x: len(np.unique(x))).rename("num_unique")], axis=1).sort_values(by=['num_unique'])
print(df1)

df2 = pd.concat([X_test.apply(lambda x: sum(x.isnull())).rename("num_missing"),
                 X_test.apply(lambda x: sum(x == 0)).rename("num_zero"),
                 X_test.apply(lambda x: len(np.unique(x))).rename("num_unique")], axis=1).sort_values(by=['num_unique'])
print(df2)

# sns.distplot(a=X_train['var_71'], rug=True,color="g")

# sns.distplot(a=X_train['var_131'], rug=True)


# # create a function which makes the plot:
# from matplotlib.ticker import FormatStrFormatter


def visualize_numeric(ax1, ax2, ax3, df, col, target):
    # plot histogram:
    df.hist(column=col, ax=ax1, bins=200)
    ax1.set_xlabel('Histogram')

    # plot box-whiskers:
    df.boxplot(column=col, by=target, ax=ax2)
    ax2.set_xlabel('Transactions')

    # plot top 10 counts:
    cnt = df[col].value_counts().sort_values(ascending=False)
    cnt.head(10).plot(kind='barh', ax=ax3)
    ax3.invert_yaxis()  # labels read top-to-bottom
# ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) #somehow not
# working
    ax3.set_xlabel('Count')


for col in list(df1.index[:20]):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    ax11 = plt.subplot(1, 3, 1)
    ax21 = plt.subplot(1, 3, 2)
    ax31 = plt.subplot(1, 3, 3)
    fig.suptitle('Feature: %s' % col, fontsize=5)
    visualize_numeric(ax11, ax21, ax31, raw_train, col, 'target')
    plt.tight_layout()

plt.show()
# corr_mat.to_csv('prob_based_cormat.csv', index=True, float_format='%.6f')
# new_df.to_csv('prob_by_var.csv', index=False, float_format='%.6f')
