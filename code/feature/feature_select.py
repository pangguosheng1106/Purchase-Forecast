# coding: utf-8

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

if __name__ == "__main__":
    feature_dir = '../../output/feature'
    train_data = pd.read_csv('%s/train_data.csv' % feature_dir, header=0)
    # V1删除大于16的用户 & V7删除大于40或小于-20的用户 & V18删除大于30的用户  
    train_data = train_data[ (train_data['V1'] < 16) &  (train_data['V7'] < 40) &  (train_data['V7'] > -20)   & (train_data['V18'] < 30) ] 
    train_X = train_data.drop(['USRID', 'FLAG', 'V6_5'], axis=1)
    train_y = train_data['FLAG']

    print("Dimension of train data: ", train_X.shape)
    parameters = {
        'gamma': [0]
    }
    xgb_model = xgb.XGBClassifier(
        learning_rate = 0.04,
        n_estimators = 800,
        max_depth = 2,
        min_child_weight = 14,
        subsample = 0.7,
        colsample_bytree = 0.7,
        nthread = 4,
        scale_pos_weight = 12,
        seed=123)
    clf = GridSearchCV(xgb_model, parameters, scoring='roc_auc', n_jobs=1, cv=5)
    clf.fit(train_X, train_y)
    for score in clf.grid_scores_:
        print(score)
    print("Best score: ", clf.best_score_)
    print("Best param: ", clf.best_params_)
    feature_names = train_X.columns
    feautre_importance = clf.best_estimator_.feature_importances_
    feature_importance_df = pd.DataFrame({'feature':feature_names, 'importance':feautre_importance})
    feature_importance_df_sorted = feature_importance_df.sort_values(by=['importance'], ascending=False)
    feature_importance_df_sorted['cumulative_importance'] = np.cumsum(feature_importance_df_sorted['importance'])
    feature_keep_condition = feature_importance_df_sorted['importance'] > 0.00
    keep_feature = feature_importance_df_sorted[feature_keep_condition].head(78)
    keep_feature.to_csv("%s/feature_selected.csv" % feature_dir, header=True, index=False)

