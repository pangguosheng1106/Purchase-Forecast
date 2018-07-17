# coding: utf-8

import gc
import sys
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

v6_threshold = 5  # v6划分的区间数
user_record_counter_threshold = 2400  # 用户记录数阈值，超过该阈值的删除
module_B_counter_threshold = 5  # module b 均值+n倍标准差的删除
module_C_counter_threshold = 5  # module c 均值+n倍标准差的删除
module_C_user_top = 100  # module_C 热门的top n 留下
module_B_user_top = 50  # module_B 热门的top n 留下
day_mean_behavior_times_threshold = 150  # 日均行为次数超过该阈值的删除

# 读取数据
input_dir = '../../input'
output_dir = '../../output/feature'
train_agg =  pd.read_csv('%s/train_agg.csv' % input_dir, sep='\t')
train_log =  pd.read_csv('%s/train_log.csv' % input_dir, sep='\t', parse_dates = ['OCC_TIM'])
train_flag =  pd.read_csv('%s/train_flg.csv' % input_dir, sep='\t')
test_agg =  pd.read_csv('%s/test_agg.csv' % input_dir, sep='\t')
test_log =  pd.read_csv('%s/test_log.csv' % input_dir, sep='\t', parse_dates = ['OCC_TIM'])

# 训练数据
train_log['module_A'], train_log['module_B'], train_log['module_C']  = train_log['EVT_LBL'].str.split('-').str
train_log['day'] = train_log['OCC_TIM'].map(lambda x:x.day)
train_log['hour'] = train_log['OCC_TIM'].map(lambda x:x.hour)
train_log['min'] =  train_log['OCC_TIM'].map(lambda x:x.minute)
train_log['sec'] =  train_log['OCC_TIM'].map(lambda x:x.second)

# 测试数据
test_log['module_A'], test_log['module_B'], test_log['module_C']  = test_log['EVT_LBL'].str.split('-').str
test_log['day'] = test_log['OCC_TIM'].map(lambda x:x.day)
test_log['hour'] = test_log['OCC_TIM'].map(lambda x:x.hour)
test_log['min'] =  test_log['OCC_TIM'].map(lambda x:x.minute)
test_log['sec'] =  test_log['OCC_TIM'].map(lambda x:x.second)

# agg变量划分离散型和连续型。训练和测试通用
discretes = []
continues = []
for col in train_agg.columns.drop('USRID'):
    num_value = len(train_agg[col].unique())
    train_col_values = set(train_agg[col].unique())
    test_col_values = set(test_agg[col].unique())
    b = test_col_values.issubset(train_col_values)
    if num_value < 100 and b:
        discretes.append(col)
    else:
        continues.append(col) 


# 对agg中的离散型变量进行独热编码。训练和测试通用
def trans_agg_discretes(df, discretes_cols):
    for col in discretes_cols:
        unique_values = df[col].unique()
        col_map = dict(zip(unique_values, range(1, len(unique_values)+1)))
        df[col] = df[col].apply(lambda x: str(col_map[x]))
    one_hot_feature = pd.get_dummies(df[discretes_cols])
    df = df.merge(one_hot_feature, left_index=True, right_index=True).drop(discretes_cols, axis=1)   
    return df 

# 对agg中的V6进行离散化。训练和测试通用
def cut_v6(df, max_value, min_value, col):
    bins = np.linspace(min_value, max_value, v6_threshold) 
    df[col] = np.digitize(df[col], bins=bins)
    df = pd.merge(df, pd.get_dummies(df[col], prefix=col), left_index=True, right_index=True)
    del df[col]
    return df

# 用户行为记录数。训练特征提取
def generate_num_user_record(df):
    user_record_counter = df.groupby(['USRID'], as_index=False)['TCH_TYP'].count()
    condition = user_record_counter['TCH_TYP'] < user_record_counter_threshold  
    outlier_user = user_record_counter[condition.apply(lambda x: not x)]['USRID'].values
    user_record_counter = user_record_counter[condition]
    user_record_counter.rename(columns={'TCH_TYP':'num_record'}, inplace=True)
    return user_record_counter, outlier_user

# 用户行为记录数。测试特征提取
def generate_num_user_record_test(df):
    user_record_counter = df.groupby(['USRID'], as_index=False)['TCH_TYP'].count()
    user_record_counter.rename(columns={'TCH_TYP':'num_record'}, inplace=True)
    return user_record_counter

# 用户点击的模块A的历史去重数。训练和测试通用
def generate_user_module_A_counter(df):
    user_module_A_counter = df.groupby(['USRID'], as_index=False)['module_A'].agg({'num_module_A':lambda x:len(set(x))})
    return user_module_A_counter

# 用户点击的模块B的历史去重数。训练和测试通用
def generate_user_module_B_counter(df):
    user_module_B_counter = df.groupby(['USRID'], as_index=False)['module_B'].agg({'num_module_B':lambda x:len(set(x))})
    mean_v = np.mean(user_module_B_counter['num_module_B'])
    std_v = np.std(user_module_B_counter['num_module_B'])
    condition = user_module_B_counter['num_module_B'] < (mean_v + module_B_counter_threshold * std_v)
    outlier_user = user_module_B_counter[condition.apply(lambda x: not x)]['USRID'].values
    user_module_B_counter = user_module_B_counter[condition]
    return user_module_B_counter, outlier_user

# 用户点击的模块C的历史去重数。训练特征提取
def generate_user_module_C_counter(df):
    user_module_C_counter = df.groupby(['USRID'], as_index=False)['module_C'].agg({'num_module_C':lambda x:len(set(x))})
    mean_v = np.mean(user_module_C_counter['num_module_C'])
    std_v = np.std(user_module_C_counter['num_module_C'])
    condition = user_module_C_counter['num_module_C'] < (mean_v + module_C_counter_threshold * std_v)  
    outlier_user = user_module_C_counter[condition.apply(lambda x: not x)]['USRID'].values
    user_module_C_counter = user_module_C_counter[condition]
    return user_module_C_counter, outlier_user

# 用户点击的模块C的历史去重数。测试特征提取
def generate_user_module_C_counter_test(df):
    user_module_C_counter = df.groupby(['USRID'], as_index=False)['module_C'].agg({'num_module_C':lambda x:len(set(x))})
    return user_module_C_counter

# 用户对每个模块A的历史点击次数。训练和测试通用
def generate_module_A_element(df):
    module_A_element = df.groupby(['USRID', 'module_A']).size().unstack()
    module_A_element.columns = ["module_A_%s" % col for col in module_A_element.columns]
    module_A_element['USRID']  = module_A_element.index
    return module_A_element

#取出热门的module_C 
module_C_user_count_sorted = train_log.groupby(['module_C'], as_index=False)['USRID'].count().sort_values(['USRID'], ascending=False)
module_C_user_count_sorted = module_C_user_count_sorted.head(module_C_user_top) 
keep_module_C = module_C_user_count_sorted['module_C'].values

#取出热门的module_B
module_B_user_count_sorted = train_log.groupby(['module_B'], as_index=False)['USRID'].count().sort_values(['USRID'], ascending=False)
module_B_user_count_sorted = module_B_user_count_sorted.head(module_B_user_top) 
keep_module_B = module_B_user_count_sorted['module_B'].values

# 用户对每个热门的模块B的历史点击次数。训练和测试通用
def generate_module_B_element(df, keep_module_B):
    keep_module_B_condiction = df['module_B'].isin(keep_module_B)
    module_B_element = df[keep_module_B_condiction].groupby(['USRID', 'module_B']).size().unstack()
    module_B_element.columns = ["module_B_%s" % col for col in module_B_element.columns]
    module_B_element['USRID']  = module_B_element.index
    return module_B_element

# 用户对每个热门的模块C的历史点击次数。训练和测试通用
def generate_module_C_element(df, keep_module_C):
    keep_module_C_condiction = df['module_C'].isin(keep_module_C)
    module_C_element = df[keep_module_C_condiction].groupby(['USRID', 'module_C']).size().unstack()
    module_C_element.columns = ["module_C_%s" % col for col in module_C_element.columns]
    module_C_element['USRID']  = module_C_element.index
    return module_C_element


# 用户日均打开小时数。训练特征使用
def generate_day_mean_open_for_train(df):
    d_group = df.groupby(['USRID', 'day'], as_index=False)['hour'].agg({'day_mean_open':lambda x:len(set(x))})
    outlier_users = d_group[d_group['day_mean_open'] >8 ]['USRID'].values 
    day_mean_open = d_group[d_group['day_mean_open'] <=8 ].groupby(['USRID'], as_index=False).mean()  
    return day_mean_open, outlier_users

# 用户日均打开小时数。测试特征使用
def generate_day_mean_open_for_test(df):
    d_group = df.groupby(['USRID', 'day'], as_index=False)['hour'].agg({'day_mean_open':lambda x:len(set(x))})
    day_mean_open = d_group.groupby(['USRID'], as_index=False).mean()
    return day_mean_open

# 用户日均行为次数。训练使用
def generate_day_mean_behavior_times_for_train(df):
    day_mean_behavior_times = df.groupby(['USRID', 'day'], as_index=False)['hour'].count().groupby(['USRID'], as_index=False)['hour'].mean()
    day_mean_behavior_times.rename(columns={'hour': 'day_mean_behavior_times'}, inplace=True)
    outlier_users = day_mean_behavior_times[day_mean_behavior_times['day_mean_behavior_times'] >= day_mean_behavior_times_threshold]['USRID'].values
    condition = day_mean_behavior_times['day_mean_behavior_times'] < day_mean_behavior_times_threshold  
    day_mean_behavior_times = day_mean_behavior_times[condition]
    return day_mean_behavior_times, outlier_users

# 用户日均行为次数。测试使用
def generate_day_mean_behavior_times_for_test(df):
    day_mean_behavior_times = df.groupby(['USRID', 'day'], as_index=False)['hour'].count().groupby(['USRID'], as_index=False)['hour'].mean()
    day_mean_behavior_times.rename(columns={'hour': 'day_mean_behavior_times'}, inplace=True)
    return day_mean_behavior_times

user_exist_day_train = train_log[['USRID', 'day']].drop_duplicates(['USRID', 'day'])
user_exist_day_test = test_log[['USRID', 'day']].drop_duplicates(['USRID', 'day'])

def number_continuous_max(number_list):
    res = []
    count = 1
    length = len(number_list)
    i = 0
    while i < length-1:
        if number_list[i] == number_list[i+1]:
            count += 1
            res.append(count)
        else:
            count = 1
        i += 1
    if res:
        return max(res)
    else:
        return 1
    
# 用户出现行为的最长连续天数。训练和测试通用
def generate_user_continuou_days(user_exist_day):
    USRIDS = []
    continuou_days = [] 
    for USRID, group_value in user_exist_day.groupby('USRID'):
        diff_value = group_value.sort_values(['day'])['day'].diff().values
        USRIDS.append(USRID)
        continuou_days.append(number_continuous_max(diff_value))
    user_continuou_days = pd.DataFrame({'USRID': USRIDS, 'continuou_days': continuou_days})
    del USRIDS
    del continuou_days
    return user_continuou_days

def set_active(ratio_days_in_month):
    if ratio_days_in_month <= 0.2:
        return "E"
    elif ratio_days_in_month > 0.2 and ratio_days_in_month <= 0.4:
        return "D"
    elif ratio_days_in_month > 0.4 and ratio_days_in_month <= 0.6:
        return "C"
    elif ratio_days_in_month > 0.6 and ratio_days_in_month <= 0.8:
        return "B"
    else:
        return "A"

# 用户月活跃率。训练和测试通用
def generate_ratio_days_in_month(user_exist_day):    
    ratio_days_in_month = user_exist_day.groupby(['USRID'], as_index=False)['day'].count()
    ratio_days_in_month.rename(columns={'day': 'day_in_month'}, inplace=True)
    ratio_days_in_month['ratio_days_in_month'] = ratio_days_in_month['day_in_month'] / 31
    ratio_days_in_month['active'] = ratio_days_in_month['ratio_days_in_month'].apply(set_active)
    return pd.get_dummies(ratio_days_in_month)

# 用户周活跃率。训练和测试通用
def generate_ratio_days_in_week(user_exist_day):
    ratio_days_in_week = user_exist_day[user_exist_day['day'] >= 25].groupby(['USRID'], as_index=False)['day'].count()
    ratio_days_in_week.rename(columns={'day': 'num_days_in_week'}, inplace=True)
    ratio_days_in_week['ratio_days_in_week'] = ratio_days_in_week['num_days_in_week'] / 7
    num_days_to_active = {7:"A", 6: "B", 5: "C", 4: "C", 3: "D", 2: "E", 1: "E"}
    ratio_days_in_week['week_active'] = ratio_days_in_week['num_days_in_week'].apply(lambda x: num_days_to_active[x])
    return pd.get_dummies(ratio_days_in_week)

# 用户首次出现行为时间距离月末的天数。训练和测试通用
def generate_user_appear_day_first(user_exist_day):
    user_exist_day_sorted = user_exist_day.sort_values(['USRID', 'day'])
    user_exist_day_sorted['rn'] = user_exist_day_sorted['day'].groupby(user_exist_day_sorted['USRID']).rank(method='min')
    user_appear_day_first = user_exist_day_sorted[(user_exist_day_sorted['rn'] == 1)]
    user_appear_day_first['first_day_gap'] = 31 - user_appear_day_first['day']
    del user_appear_day_first['rn']
    del user_appear_day_first['day']
    return user_appear_day_first

# 用户最后一次出现行为距离月末的天数。训练和测试通用
def generate_user_appear_day_last(user_exist_day):
    user_exist_day_sorted = user_exist_day.sort_values(['USRID', 'day'], ascending=False)
    user_exist_day_sorted['rn'] = user_exist_day_sorted['day'].groupby(user_exist_day_sorted['USRID']).rank(method='min', ascending=False)
    user_appear_day_last = user_exist_day_sorted[(user_exist_day_sorted['rn'] == 1)].drop(['rn'], axis=1)
    user_appear_day_last['last_day_gap'] = 31 - user_appear_day_last['day']
    del user_appear_day_last['day']
    return user_appear_day_last

# 用户行为间隔天数的统计量。训练和测试通用
def generate_behavior_gap_statistical_data(user_exist_day):
    behavior_gap_statistical = []
    for USRID, group_value in user_exist_day.groupby('USRID'):
        diff_value = group_value.sort_values(['day'])['day'].diff().values
        diff_value = list(diff_value)
        # 只有一天的数据，直接nan
        if len(diff_value) == 1:
            mean_v = np.nan
            median_v = np.nan
            std_v = np.nan
        else:
            mean_v = np.nanmean(diff_value)
            median_v = np.nanmedian(diff_value)
            std_v = np.nanstd(diff_value)
        behavior_gap_statistical.append((USRID, mean_v, median_v, std_v))
    behavior_gap_statistical_data = pd.DataFrame(behavior_gap_statistical, columns=['USRID', 'behavior_gap_mean', 'behavior_gap_median', 'behavior_gap_std'])
    del behavior_gap_statistical
    return behavior_gap_statistical_data

# 不同操作类型的历史次数。训练和测试通用
def generate_operation_type(df):
    operation_type = df.groupby(['USRID', 'TCH_TYP']).size().unstack()
    for col in operation_type.columns:
        operation_type.rename(columns={col: "TCH_TYP_%s" % col}, inplace=True)
    operation_type.fillna(0, inplace=True)
    operation_type.reset_index(inplace=True)
    return operation_type


# 各模块两两相除。训练和测试通用
def add_module_ratio(df):
    astr = "module_A"
    bstr = "module_B"
    cstr = "module_C"
    for col in df.columns:
        if col.startswith(astr) or col.startswith(bstr) or col.startswith(cstr):
            new_col = "%s_ratio" % col
            df[new_col] = df[col] / df['num_record']
    return df


if __name__ == "__main__":

    train_agg = trans_agg_discretes(train_agg, discretes)
    test_agg = trans_agg_discretes(test_agg, discretes)
    train_agg_filter_outlier = train_agg 
    max_value = train_agg_filter_outlier['V6'].max()
    min_value = train_agg_filter_outlier['V6'].min()
    train_agg_filter_outlier = cut_v6(train_agg_filter_outlier, max_value, min_value, "V6")
    test_agg = cut_v6(test_agg, max_value, min_value, "V6")
    
    user_record_counter, outlier_user2 = generate_num_user_record(train_log)
    user_module_A_counter = generate_user_module_A_counter(train_log)
    user_module_B_counter, outlier_user3 = generate_user_module_B_counter(train_log)
    user_module_C_counter, outlier_user4 = generate_user_module_C_counter(train_log)
    module_A_element = generate_module_A_element(train_log)
    module_B_element = generate_module_B_element(train_log, keep_module_B)
    module_C_element = generate_module_C_element(train_log, keep_module_C)
    ratio_days_in_month = generate_ratio_days_in_month(user_exist_day_train)
    ratio_days_in_week = generate_ratio_days_in_week(user_exist_day_train)
    day_mean_open, outlier_users5 = generate_day_mean_open_for_train(train_log)
    day_mean_behavior_times, outlier_users6 = generate_day_mean_behavior_times_for_train(train_log)
    user_continuou_days = generate_user_continuou_days(user_exist_day_train)
    user_appear_day_first = generate_user_appear_day_first(user_exist_day_train)
    user_appear_day_last = generate_user_appear_day_last(user_exist_day_train)
    behavior_gap_statistical_data = generate_behavior_gap_statistical_data(user_exist_day_train)
    operation_type = generate_operation_type(train_log)
    
    user_record_counter_test  = generate_num_user_record_test(test_log)
    user_module_A_counter_test = generate_user_module_A_counter(test_log)
    user_module_B_counter_test = generate_user_module_B_counter(test_log)
    user_module_C_counter_test = generate_user_module_C_counter(test_log)
    module_A_element_test = generate_module_A_element(test_log)
    module_B_element_test = generate_module_B_element(test_log, keep_module_B)
    module_C_element_test = generate_module_C_element(test_log, keep_module_C)
    ratio_days_in_month_test = generate_ratio_days_in_month(user_exist_day_test)
    ratio_days_in_week_test = generate_ratio_days_in_week(user_exist_day_test)
    day_mean_open_test = generate_day_mean_open_for_test(test_log)
    day_mean_behavior_times_test = generate_day_mean_behavior_times_for_test(test_log)
    user_continuou_days_test = generate_user_continuou_days(user_exist_day_test)
    user_appear_day_first_test = generate_user_appear_day_first(user_exist_day_test)
    user_appear_day_last_test = generate_user_appear_day_last(user_exist_day_test)
    behavior_gap_statistical_data_test = generate_behavior_gap_statistical_data(user_exist_day_test)
    operation_type_test = generate_operation_type(test_log)
    
    train_data = pd.merge(train_flag, user_record_counter, on=['USRID'], how='left')
    train_data = pd.merge(train_data, user_module_A_counter, on=['USRID'], how='left')
    train_data = pd.merge(train_data, user_module_B_counter, on=['USRID'], how='left')
    train_data = pd.merge(train_data, user_module_C_counter, on=['USRID'], how='left')
    train_data = pd.merge(train_data, module_A_element, on=['USRID'], how='left')
    train_data = pd.merge(train_data, module_B_element, on=['USRID'], how='left')
    train_data = pd.merge(train_data, module_C_element, on=['USRID'], how='left')
    train_data = pd.merge(train_data, ratio_days_in_month, on=['USRID'], how='left')
    train_data = pd.merge(train_data, ratio_days_in_week, on=['USRID'], how='left')
    train_data = pd.merge(train_data, day_mean_open, on=['USRID'], how='left')
    train_data = pd.merge(train_data, day_mean_behavior_times, on=['USRID'], how='left')
    train_data = pd.merge(train_data, user_continuou_days, on=['USRID'], how='left')
    train_data = pd.merge(train_data, user_appear_day_first, on=['USRID'], how='left')
    train_data = pd.merge(train_data, user_appear_day_last, on=['USRID'], how='left')
    train_data = pd.merge(train_data, operation_type, on=['USRID'], how='left')
    train_data = pd.merge(train_data, behavior_gap_statistical_data, on=['USRID'], how='left')
    train_data = pd.merge(train_data, train_agg_filter_outlier, on=['USRID'], how='left')
    
    del user_record_counter,user_module_A_counter,user_module_B_counter,user_module_C_counter
    del module_A_element,module_B_element,module_C_element,ratio_days_in_month,ratio_days_in_week,day_mean_open
    del day_mean_behavior_times,user_continuou_days,user_appear_day_first,user_appear_day_last,operation_type
    del behavior_gap_statistical_data,train_agg_filter_outlier, train_agg, train_log
    
    test_data = pd.merge(test_agg, user_record_counter_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, user_module_A_counter_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, user_module_B_counter_test[0], on=['USRID'], how='left')
    test_data = pd.merge(test_data, user_module_C_counter_test[0], on=['USRID'], how='left')
    test_data = pd.merge(test_data, module_A_element_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, module_B_element_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, module_C_element_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, ratio_days_in_month_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, ratio_days_in_week_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, day_mean_open_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, day_mean_behavior_times_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, user_continuou_days_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, user_appear_day_first_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, user_appear_day_last_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, behavior_gap_statistical_data_test, on=['USRID'], how='left')
    test_data = pd.merge(test_data, operation_type_test, on=['USRID'], how='left')

    del user_record_counter_test,user_module_A_counter_test,user_module_B_counter_test,user_module_C_counter_test
    del module_A_element_test, module_B_element_test, module_C_element_test, ratio_days_in_month_test
    del ratio_days_in_week_test, day_mean_open_test, day_mean_behavior_times_test, user_continuou_days_test
    del user_appear_day_first_test, user_appear_day_last_test, behavior_gap_statistical_data_test, operation_type_test

    #保存特征数据
    train_data.to_csv('%s/train_data.csv' % output_dir, header=True, index=False)
    test_data.to_csv('%s/test_data.csv' % output_dir, header=True, index=False)
