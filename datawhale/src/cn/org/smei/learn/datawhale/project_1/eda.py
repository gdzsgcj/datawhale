# -*- coding: utf-8 -*-
'''
Created on 2019年5月11日

@author: gdgzgcj
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10,6
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
pd.set_option('display.max_column',None)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

DATA_FILE = '../dataset/data.csv'


def read_preview_data():
    """
    本方法主要用于数据预览,类型分析等
    """
    df = pd.read_csv(DATA_FILE,encoding='gbk')
    print(df.head(3))
    # 查看数据类型和缺失情况
    print(df.info())
    # 部分数据存在缺失 数据类型分布 float64(70), int64(13), object(7)
    print(df.shape)
    # 共计4754行,90个维度
    # 返回数据
    return df

def feature_analysis_clear(df):
    """
    本方法主要用于特征清理
    经过初步分析,Unnamed: 0,custid,trade_no,bank_card_no,id_name 这些列都是基于顺序号或者姓名等
    基本可以判断这些列与最终结果无关,所以可以先将这些列删除
    """
    drop_col = ['Unnamed: 0','custid','trade_no','bank_card_no','id_name']
    df.drop(columns=drop_col,axis=1,inplace=True)
    # source 该列只有一个特征,对于分析也无任何意义,删除
    df.drop('source',axis=1,inplace=True)
    # 查看下个列的方差情况,低方差列过滤
    print(df.std().sort_values())
    # 通过方差对比,也能发现部分特征没有多少用
    # 通过热力图，查看各列的相关性
#     plt.figure(figsize=(10,8))
#     sns.heatmap(df.corr())
#     plt.show()
    # 返回最后删除无关特征后的数据
    return df

def transform_datatype(df):
    """
    本方法的主要作用是对数据类型进行转换
    """
    # 先预览一下经过清理后主要纯在的数据类型
    print(df.info())
    # 当前数据中，存在3个对象类型,其实主要是三个日期类型和一个分类类型 下面将对他们进行转换
    # first_transaction_time,latest_query_time,loans_latest_time 这三个需要转换为日期类型
    # reg_preference_for_trad 这个是分类类型,要将它转换为离散化变量
    df['first_transaction_time'] = pd.to_datetime(df['first_transaction_time'])
    df['latest_query_time'] = pd.to_datetime(df['latest_query_time'])
    df['loans_latest_time'] = pd.to_datetime(df['loans_latest_time'])
    print(df.info())
    # 查看一下reg_preference_for_trad 数据分布情况
    print(df['reg_preference_for_trad'].value_counts())
    # 查看是否有缺失值
    print(df['reg_preference_for_trad'].isnull().sum())
    # 缺失值为2,缺少数量很少,在相对于总样本数4754 占比非常小,可以考虑删除
    # 同时观察该原始数据,该项空缺，别的列很多也空缺,所以直接删除
    # 如果要填充的话，这儿采取中位数填充
    df = df.dropna(subset=['reg_preference_for_trad'])
    print(df['reg_preference_for_trad'].isnull().sum())
    print(df['reg_preference_for_trad'].value_counts())
    # 对该列进行编码
    col_map = {'一线城市':1, '二线城市':2,'三线城市':3,'其他城市':4,'境外':0}
    df['reg_preference_for_trad_new'] = df['reg_preference_for_trad'].map(col_map)
    print(df['reg_preference_for_trad_new'].head(5))
    # 删除原来的列
    df.drop(['reg_preference_for_trad'],axis=1,inplace=True)
    return df


def main():
    """第一节：EDA 探索性数据分析"""
    """1.1:读取和预览数据"""
    df = read_preview_data()
    """1.2:无关特征删除"""
    df = feature_analysis_clear(df)
    """1.3:数据类型转换"""
    df = transform_datatype(df)
    
    
if __name__ == '__main__':
    main()