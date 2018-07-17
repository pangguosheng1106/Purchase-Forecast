# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

"""
线下cv 0.86088
"""

M3 = xgb.XGBClassifier(
    learning_rate = 0.03,
    n_estimators = 375,
    max_depth = 5,
    min_child_weight = 6,
    subsample = 0.7,
    colsample_bytree = 0.7,
    gamma = 0.3,
    nthread = 4,
    seed=123)

if __name__ == "__main__":
    # 加载特征数据
    feature_dir = '../../output/feature'
    output_dir = '../../output/submission'
    train_data = pd.read_csv('%s/train_data.csv' % feature_dir, header=0)
    test_data = pd.read_csv('%s/test_data.csv' % feature_dir, header=0)
    feature_selected = pd.read_csv('%s/feature_selected.csv' % feature_dir, header=0)
    keep_feature = feature_selected['feature'].values

    # 生成训练数据、label数据
    train_data = train_data[ (train_data['V1'] < 16) &  (train_data['V7'] < 40) &  (train_data['V7'] > -20)   & (train_data['V18'] < 30) ]    
    train_X = train_data.drop(['USRID', 'FLAG'], axis=1)[keep_feature]
    train_y = train_data['FLAG']
    print("Dimension of train data: ", train_X.shape)

    # 训练
    parameters = {'gamma': [0.3]}
    clf_M3= GridSearchCV(M3, parameters, scoring='roc_auc', n_jobs=1, cv=5)
    clf_M3.fit(train_X, train_y)
    for score in clf_M3.grid_scores_:
        print(score)
    print("Best score: ", clf_M3.best_score_)
    print("Best param: ", clf_M3.best_params_)

    # 预测、保存
    submit_M3 = test_data[['USRID']]
    t = test_data.drop(['USRID'], axis=1)[keep_feature]
    submit_M3['RST']  = clf_M3.predict_proba(t)[:, 1]
    submit_M3.to_csv('%s/submit_M3.csv' % output_dir, index=None,sep='\t')
