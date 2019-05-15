# -*- coding: utf-8 -*-
'''
Created on 2019年5月14日
特征工程

通过IV值进行特征选择,代码学习于https://blog.csdn.net/weixin_41710583/article/details/86027573
@author: gdgzg
'''
import pandas as pd
import numpy as np
import warnings
import time
import math
warnings.filterwarnings('ignore')
pd.set_option('display.max_column',None)
DATA_FILE = '../dataset/data1.csv'

from sklearn.utils.multiclass import type_of_target
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

def load_data():
    """
    加载数据
    """
    df = pd.read_csv(DATA_FILE,parse_dates=['first_transaction_time','latest_query_time','loans_latest_time'])
    print(df.head(5))
    return df

def create_field_by_df(df):
    """
    通过原来的数据,合并建立一些新的列
    """
    # max_consume_count_later_6_month 和jewelry_consume_count_last_6_month 合并，昨晚6个月总消费次数
    df['all_consume_count_later_6_month'] = df['max_consume_count_later_6_month'] + df['jewelry_consume_count_last_6_month']
    """
    trans_top_time_last_1_month     上月交易时长
    trans_top_time_last_6_month    上6个月交易时长
    consume_top_time_last_1_month    上月消费最高耗时
    consume_top_time_last_6_month    上六个月消费最高耗时
    """
    # 将上面的交易和消费类别分别累计统计
    df['trans_consume_top_time_last_1_month'] = df['trans_top_time_last_1_month'] + df['consume_top_time_last_1_month']
    df['trans_consume_time_last_6_month'] = df['trans_top_time_last_6_month'] + df['consume_top_time_last_6_month']
    
    """
    query_org_count        组织查询次数
    query_finance_count    金融查询次数
    query_cash_count    现金查询次数
    query_sum_count        
    """
    # 查询次数汇总
    df['all_query_count'] = df['query_org_count'] + df['query_finance_count'] + df['query_cash_count'] + df['query_sum_count']
    # 通过贷款次数，贷款已经结清次数，贷款逾期次数，可以计算出贷款还未偿还次数
    """
    loans_count    贷款次数
    loans_settle_count    贷款结算次数
    loans_overdue_count    贷款逾期次数
    """
    df['loans_undone_count'] = df['loans_count'] - df['loans_settle_count'] - df['loans_overdue_count']
    
    # 下面的贷款信用和消费信用，可以累计起来昨晚总的信用和限额
    """
    loans_credit_limit    贷款信用限额  consfin_credit_limit    消费贷款信用额度
    loans_max_limit        贷款最高限额  consfin_max_limit    消费最大限额
    loans_avg_limit        贷款平均限额  consfin_avg_limit    消费平均限额
    """
    df['loans_consfin_credit_limit'] = df['loans_credit_limit'] + df['consfin_credit_limit']
    df['loans_consfin_max_limit'] = df['loans_max_limit'] + df['consfin_max_limit']
    df['loans_consfin_avg_limit'] = df['loans_avg_limit'] + df['consfin_avg_limit']
    return df



def discrete(x):
    """
    对数据进行离散化
    """
    res = np.zeros(x.shape)
    # 这儿使用5等分进行离散化,在实际环境中要看数据的分布了
    for i in range(5):
        point1 = stats.scoreatpercentile(x, i * 20)
        point2 = stats.scoreatpercentile(x, (i + 1) * 20)
        x1 = x[np.where((x >= point1) & (x <= point2))]
        mask = np.in1d(x,x1)
        res[mask] = i + 1 # 将[i, i+1]块内的值标记成i+1
    return res 
 
def woe_single_x(x ,y ,feature, event=1):
    """
    求单个特征的woe值
    """ 
    event_total = sum(y==event)
    non_event_total = y.shape[-1] - event_total
    iv = 0;
    woe_dict = {}
    for x1 in set(x):
        y1 = y.reindex(np.where(x == x1)[0])
        event_count = sum(y1 == event)
        non_event_count = y1.shape[-1] - event_count
        rate_event = event_count / event_total
        rate_non_event = non_event_count / non_event_total
        
        if rate_event == 0:
            rate_event = 0.0001
        elif rate_non_event == 0:
            rate_non_event = 0.0001
        woei = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woei
        iv += (rate_event - rate_non_event) * woei
    return woe_dict,iv
 
def woe(X,y,event=1):
    res_woe = [] 
    iv_dict = {}
    for feature in X.columns:
        x = X[feature].values
        # 判断x 是否为连续变量,如果是,就要进行离散化
        if type_of_target(x) == 'continuous':
            x = discrete(x)
        woe_dict,iv = woe_single_x(x, y, feature, event)
        iv_dict[feature] = iv
        res_woe.append(woe_dict)
     
    return iv_dict
        
   
def information_value_select(df):
    """
        通过IV值进行特征选择
    """   
    X = df.drop(['status','first_transaction_time','latest_query_time','loans_latest_time'],axis=1)
    y = df['status']
    iv_dict = woe(X,y)
    dict = sorted(iv_dict.items(), key = lambda x : x[1], reverse = True)
    print('Information Value:\n','*' * 40)
    # iv 0.02以上才有价值
    retain_col = ['status']
    importance_score = 0.02
    for i,v in dict:
        print('列名:',i ,', 重要性评分:',v)
        if v > importance_score:
            retain_col.append(i)
    # 这儿可以通过分值筛选出列
    print('\n有价值的列:',retain_col)
    return retain_col
    
def randomforest_select(df):
    """
    通过随机森林选择特征
    """
    X = df.drop(['status','first_transaction_time','latest_query_time','loans_latest_time'],axis=1)
    y = df['status']
    forest = RandomForestClassifier(n_jobs=-1)
    forest.fit(X,y)
    importance = forest.feature_importances_
    dict_forest = dict(zip(X.columns,importance))
    dict_forest = sorted(dict_forest.items(),key = lambda x:x[1],reverse = True)
    print('随机森林:\n','*' * 40)
    importance_score = 0.01
    retain_col = ['status']
    for i,v in dict_forest:
        print('列名:',i ,', 重要性评分:',v)
        if v > importance_score :
            retain_col.append(i)
            
    print('\n有价值的列:',retain_col)        
    return retain_col
            
def lassocv_feature_select(df):
    """
    通过LassoCV 进行特征选择
    """    
    X = df.drop(['status','first_transaction_time','latest_query_time','loans_latest_time'],axis=1)
    y = df['status']
    model_lasso = LassoCV(alphas = [0.1,1,0.001, 0.0005])
    model_lasso.fit(X,y)
    coef = pd.Series(model_lasso.coef_,index=X.columns)
    print(coef.sort_values(ascending=False))
    
    
def main():
    # 1.1 加载数据
    df = load_data()
    # 2.1 生成新的列
    df = create_field_by_df(df)
    # 2.2 通过IV值进行特征选择 iv 值>0.02才有价值
    retain_col = information_value_select(df)
    # 2.3 通过随机森林选择特征 feature_importances_ > 0.01才有价值
    randomforest_select(df)
    # 2.3 通过LassoCV进行特征选择
    lassocv_feature_select(df)
    
    # 生成新的数据集
    df_new = df[retain_col]
    df_new.to_csv('../dataset/data2.csv',index = None)
    
    
if __name__ == '__main__':
    main()