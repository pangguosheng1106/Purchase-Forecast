# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

from single_xgb_M1 import M1
from single_xgb_M2 import M2
from single_xgb_M3 import M3

"""
线下cv 0.86121
"""

if __name__ == '__main__':
    # 加载特征数据
    feature_dir = '../../output/feature'
    output_dir = '../../output/submission'
    train_data = pd.read_csv('%s/train_data.csv' % feature_dir, header=0)
    test_data = pd.read_csv('%s/test_data.csv' % feature_dir, header=0)
    feature_selected = pd.read_csv('%s/feature_selected.csv' % feature_dir, header=0)
    keep_feature = feature_selected['feature'].values

    # 生成训练数据、label数据
    train_data = train_data[ (train_data['V1'] < 16) &  (train_data['V7'] < 40) &  (train_data['V7'] > -20)   & (train_data['V18'] < 30) ] 
    X = train_data.drop(['USRID', 'FLAG'], axis=1)[keep_feature]
    y = train_data['FLAG']
    stacking_result = test_data[['USRID']]
    X_submission = test_data.drop(['USRID'], axis=1)[keep_feature]
    print("Dimension of train data: ", X.shape)

    # 部分变量初始化
    np.random.seed(4396)
    n_folds = 5
    verbose = True
    shuffle = False

    # 是否打乱数据顺序
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))                         
    models = [M1, M2, M3]

    print ("Creating train and test sets for blending.")
    dataset_blend_train = np.zeros((X.shape[0], len(models)))  # 创建一个80000行，4列的0矩阵，矩阵train
    dataset_blend_test = np.zeros((X_submission.shape[0], len(models)))  # 创建一个20000行，4列的0矩阵，矩阵test

    for j, model in enumerate(models):
        print( j, model)  # 序号，分类器
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))   # 创建一个20000行，5列的矩阵，矩阵valid
        for i, (train, test) in enumerate(skf):  # train为训练数据索引, test为验证数据索引
            print ("Fold", i)
            X_train = X.iloc[train]
            y_train = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
            model.fit(X_train, y_train)  # 使用一份数据进行模型训练
            y_submission = model.predict_proba(X_test)[:, 1]  # 对验证数据的预测概率
            dataset_blend_train[test, j] = y_submission  # 验证数据对应的行, 第j个模型对应的列，放入预测的概率
            dataset_blend_test_j[:, i] = model.predict_proba(X_submission)[:, 1]  # 对测试数据预测得到概率， 放入矩阵矩阵valid的第i列，交叉验证的第i份数据
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)   #多重训练得到的模型对测试数据预测得到多份概率的均值，放入矩阵test的j列中，对应第j模型

    # 对预测结果二次训练
    y = train_data['FLAG']
    parameters = {
        'C': [0.0003],
    }
    lr_model = LogisticRegression(
        penalty='l2',    
        C=0.5,
        max_iter=50, 
        intercept_scaling=1,
        fit_intercept=True,
        dual=False,solver='liblinear', random_state=123,multi_class='ovr', warm_start=False)

    stacking_clf = GridSearchCV(lr_model, parameters, scoring='roc_auc', n_jobs=-1, cv=5)
    stacking_clf.fit(dataset_blend_train, y)
    for score in stacking_clf.grid_scores_:
        print(score)
    print("Best score: ", stacking_clf.best_score_)
    print("Best param: ", stacking_clf.best_params_)

    y_submission = stacking_clf.predict_proba(dataset_blend_test)[:, 1]
    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    stacking_result['RST'] = y_submission
    stacking_result[['USRID', 'RST']].to_csv('%s/stacking_result.csv' % output_dir, index=None, sep='\t')

