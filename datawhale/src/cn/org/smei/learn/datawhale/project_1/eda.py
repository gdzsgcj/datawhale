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
    df = pd.read_csv(DATA_FILE,encoding='gbk',parse_dates=['first_transaction_time','latest_query_time','loans_latest_time'])
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

def feature_corr_analysis(df):
    """
    特征相关性分析
    """
#     X_train = df.drop('status', axis=1)
#     y_train = df['status']
#     
    print('\n打印相关性矩阵\n',df.corr()['status'])
    return df
    

def transform_datatype(df):
    """
    本方法的主要作用是对数据类型进行转换
    """
    # 先预览一下经过清理后主要纯在的数据类型
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
    
    # 这是经过特征工程后得出的比较重要的列,现在对这些列进行判断,包括对他们的数据进行处理
    retain_col = ['status', 'trans_fail_top_count_enum_last_1_month', 'history_fail_fee', 'apply_score', 'loans_score', 'max_cumulative_consume_later_1_month', 'latest_one_month_fail', 'loans_overdue_count', 'trans_day_last_12_month', 'repayment_capability', 'consfin_avg_limit', 'trans_amount_3_month', 'avg_price_last_12_month', 'trans_amount_increase_rate_lately', 'latest_query_day', 'loans_latest_day', 'trans_fail_top_count_enum_last_12_month', 'trans_days_interval', 'historical_trans_amount', 'trans_fail_top_count_enum_last_6_month', 'trans_activity_day', 'number_of_trans_from_2011', 'historical_trans_day', 'consfin_credit_limit', 'trans_days_interval_filter', 'apply_credibility', 'abs', 'history_suc_fee', 'latest_one_month_suc', 'loans_max_limit', 'first_transaction_day', 'loans_avg_limit', 'loans_settle_count', 'trans_top_time_last_6_month', 'pawns_auctions_trusts_consume_last_6_month', 'loans_count', 'latest_six_month_apply', 'pawns_auctions_trusts_consume_last_1_month', 'rank_trad_1_month', 'trans_activity_month']
    df = df[retain_col]
    
    return df

def data_distribution(df,col,title):
    plt.figure(figsize=(10,8))
    sns.distplot(df[col], bins=10)
    plt.title(title,fontsize=18)
    plt.show()

def data_exploratory_analysis(df):
    """
    对所有相关联的列进行探索性分析
    """
    col_list = list(df.columns)
    col_list.remove('status')
    for col in col_list:
        # 查看相关数据的缺失等情况
        print(col,'\n 数据统计: 数据集总量',len(df),'  当前数据集数据量:',df[col].describe()['count'],' 缺失率:',round((len(df)-df[col].describe()['count'])/len(df),4)*100,'%, 数据详细情况:',df[col].describe())
    
    # 分析结果     
    """
        trans_fail_top_count_enum_last_1_month 上个月交易失败次数
        数据集总量 4752,count:4738 数据存在轻微的缺失(0.29 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'trans_fail_top_count_enum_last_1_month','上月交易失败次数分布')
    # 分布图显示该数据大部分是0和1,基于该标题定义为交易失败的情况，
    # 我们就假设空缺的数据为交易正常的, 以0来进行填充
    df['trans_fail_top_count_enum_last_1_month'] = df['trans_fail_top_count_enum_last_1_month'].fillna(0)
    
    """
        history_fail_fee 历史失败费用
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'history_fail_fee','历史失败费用')
    # 从图中观察到大部分的数据都在0,说明失败的情况很少,正常的情况比较多
    # 所以对缺失值填充0    
    df['trans_fail_top_count_enum_last_1_month'] = df['trans_fail_top_count_enum_last_1_month'].fillna(0)    
    
    """
        apply_score 积分
        数据集总量 4752,count:4448 数据缺失(6.4 %), 空缺的,以交易失败次数为0来填充。
    """
    # 先看下该数据的分布情况
    data_distribution(df,'apply_score','积分')
    # 从图中观察到积分主要分布在550-650这个情况，而且成正态分布情况,
    # 这儿采用均值进行填充   
    df['apply_score'] = df['apply_score'].fillna(df['apply_score'].mean()) 
        
    """
        loans_score 贷款评分
        数据集总量 4752,count:4455 数据缺失(6.25 %), 空缺的,以交易失败次数为0来填充。
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_score','贷款评分')
    # 从图中观察到积分主要分布在550-650这个情况，而且成正态分布情况,
    # 这儿采用均值进行填充   
    df['loans_score'] = df['loans_score'].fillna(df['loans_score'].mean()) 
    
    """
        latest_one_month_fail 最近一个月失败费用
        数据集总量 4752,count:4455 数据缺失(6.25 %), 空缺的,以交易失败次数为0来填充。
    """
    # 先看下该数据的分布情况
    data_distribution(df,'latest_one_month_fail','最近一个月失败费用')
    # 从图中观察到数据主要分布在0和1,考虑标题,假设最近失败很少,
    # 采用0进行填充  
    df['latest_one_month_fail'] = df['latest_one_month_fail'].fillna(0)    
    # 该数据可能存在异常点,后面再来处理
    
    """
        loans_overdue_count 贷款逾期次数
        数据集总量 4752,count:4455 数据缺失(6.25 %), 空缺的,以交易失败次数为0来填充。
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_overdue_count','贷款逾期次数')
    # 从图中观察到数据主要分布在0到3,这儿采用众数来填充,
    # 
    df['loans_overdue_count'] = df['loans_overdue_count'].fillna(df['loans_overdue_count'].median())    
    # 该数据可能存在异常点,后面再来处理    
    
    """
        consfin_avg_limit 消费平均限额
        数据集总量 4752,count:4455 数据缺失(6.25 %), 空缺的,以交易失败次数为0来填充。
    """
    # 先看下该数据的分布情况
    data_distribution(df,'consfin_avg_limit','消费平均限额')
    # 该数据大部分集中在0-8000,差别还比较大,均值在8000，
    # 暂时定采用均值填充
    df['consfin_avg_limit'] = df['consfin_avg_limit'].fillna(df['consfin_avg_limit'].mean())    
    # 该数据可能存在异常点,后面再来处理    
 
    """
        trans_amount_increase_rate_lately 最近交易增长比例
        数据集总量 4752,count:4751 数据缺失(0.02 %),
    """
    # 先看下该数据的分布情况
    data_distribution(df,'trans_amount_increase_rate_lately','最近交易增长比例')
    # 只有一行空缺
    # 采用0填充
    df['trans_amount_increase_rate_lately'] = df['trans_amount_increase_rate_lately'].fillna(0)    
    # 该数据可能存在异常点,后面再来处理           
    
    """
        latest_query_day 最新查询天数
        数据集总量 4752,count:4455 数据缺失(6.4 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'latest_query_day','最新查询天数')
    # 大部分的查询天数都在2天-3天
    # 暂时定3天
    df['latest_query_day'] = df['latest_query_day'].fillna(3)    
    # 该数据可能存在异常点,后面再来处理        

    """
        loans_latest_day 最近贷款天数
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_latest_day','最近贷款天数')
    # 采用众数填充
    df['loans_latest_day'] = df['loans_latest_day'].fillna(df['loans_latest_day'].median())    
    # 该数据可能存在异常点,后面再来处理    
    
    """
        trans_fail_top_count_enum_last_12_month 上12个月交易失败次数
        数据集总量 4752,count:4738 数据缺失(0.29 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'trans_fail_top_count_enum_last_12_month','上12个月交易失败次数')
    # 采用均值填充
    df['trans_fail_top_count_enum_last_12_month'] = df['trans_fail_top_count_enum_last_12_month'].fillna(df['trans_fail_top_count_enum_last_12_month'].mean())    
    # 该数据可能存在异常点,后面再来处理        

    """
        trans_fail_top_count_enum_last_6_month 上六个月交易失败次数
        数据集总量 4752,count:4738 数据缺失(0.29 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'trans_fail_top_count_enum_last_6_month','上六个月交易失败次数')
    # 采用均值填充
    # 采用50%分位数填充
    val_50_cur = np.percentile(df['trans_fail_top_count_enum_last_6_month'],0.5)
    df['trans_fail_top_count_enum_last_6_month'] = df['trans_fail_top_count_enum_last_6_month'].fillna(val_50_cur)    
    # 该数据可能存在异常点,后面再来处理            
    
    """
        consfin_credit_limit 消费贷款信用额度
        数据集总量 4752,count:4738 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'consfin_credit_limit','消费贷款信用额度')
    # 该数据大部分分布在8000内,右偏比较严重
    # 采用50%分位数填充
    val_50_cur = np.percentile(df['consfin_credit_limit'],0.5)
    df['consfin_credit_limit'] = df['consfin_credit_limit'].fillna(val_50_cur)    
    # 该数据可能存在异常点,后面再来处理    

    
    """
        trans_days_interval_filter 交易天数间隔过滤
        数据集总量 4752,count:4738 数据缺失(0.13 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'trans_days_interval_filter','交易天数间隔过滤')
    # 直接填充0
    df['trans_days_interval_filter'] = df['trans_days_interval_filter'].fillna(0)    
    # 该数据可能存在异常点,后面再来处理    
    
    """
        apply_credibility 客户信用分
        数据集总量 4752,count:4448 数据缺失(6.4 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'apply_credibility','客户信用分')
    # 根据标题属性和数据分布大概情况,采用均值填充
    df['apply_credibility'] = df['apply_credibility'].fillna(df['apply_credibility'].mean())    
    # 该数据可能存在异常点,后面再来处理            

    
    """
        history_suc_fee 历史成功费用
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'history_suc_fee','历史成功费用')
    # 该数据大部分分布在0到50,右偏比较严重
    # 采用50%分位数填充
    val_50_cur = np.percentile(df['history_suc_fee'],0.5)
    df['history_suc_fee'] = df['history_suc_fee'].fillna(val_50_cur)   
    # 该数据可能存在异常点,后面再来处理  

    """
        latest_one_month_suc 最近一个月成功费用
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'latest_one_month_suc','最近一个月成功费用')
    # 大部分是0
    # 直接0填充
    df['latest_one_month_suc'] = df['latest_one_month_suc'].fillna(0)   
    # 该数据可能存在异常点,后面再来处理  

    """
        loans_max_limit 贷款最高限额
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_max_limit','贷款最高限额')
    # 从标题属性理解,采用均值填充
    df['loans_max_limit'] = df['loans_max_limit'].fillna(df['loans_max_limit'].mean())   
    # 该数据可能存在异常点,后面再来处理  

    """
        loans_avg_limit 贷款平均限额
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_avg_limit','贷款平均限额')
    # 从标题属性理解,采用均值填充
    df['loans_avg_limit'] = df['loans_avg_limit'].fillna(df['loans_avg_limit'].mean())   
    # 该数据可能存在异常点,后面再来处理  


    """
        loans_settle_count 贷款结算次数
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_settle_count','贷款结算次数')
    # 从标题属性理解,缺失值以众数填充
    df['loans_settle_count'] = df['loans_settle_count'].fillna(df['loans_settle_count'].median())   
    # 该数据可能存在异常点,后面再来处理  

    """
        trans_top_time_last_6_month 上6个月交易时长
        数据集总量 4752,count:4455 数据缺失(0.13 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'trans_top_time_last_6_month','上6个月交易时长')
    # 缺失数量很小,以0填充
    df['trans_top_time_last_6_month'] = df['trans_top_time_last_6_month'].fillna(0)   
    # 该数据可能存在异常点,后面再来处理 

    """
        loans_count 贷款次数
        数据集总量 4752,count:4455 数据缺失(6.25 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'loans_count','贷款次数')
    # 从属性的意思和数据分布情况来看,采用均值填充
    df['loans_count'] = df['loans_count'].fillna(df['loans_count'].mean())   
    # 该数据可能存在异常点,后面再来处理 

    """
        latest_six_month_apply 最近六个月申请次数
        数据集总量 4752,count:4448 数据缺失(6.4 %)
    """
    # 先看下该数据的分布情况
    data_distribution(df,'latest_six_month_apply','最近六个月申请次数')
    # 从属性的意思和数据分布情况来看,采用众数填充
    df['latest_six_month_apply'] = df['latest_six_month_apply'].fillna(df['latest_six_month_apply'].median())   
    # 该数据可能存在异常点,后面再来处理 
    return df
    
def missing_value_processing(df):
    """
    本方法主要用于缺失值处理
    """
    # 首先查看缺失值分布情况,先对缺失值进行统计
    df_miss = df.isnull().sum().sort_values()
    # 统计出存在缺失的数据
    df_miss = df_miss[df_miss.values>0]
    print('缺失值分布:\n',df_miss)
    # 共计有57个列存在缺失情况
    # 删除缺失值在24以下的全部特征 占总数量比不足1%的数据
    dropna_col = list(df_miss[df_miss.values<25].index)
    df = df.dropna(subset=dropna_col)
    # 观察该列 avg_price_top_last_12_valid_month 发现数据偏差不大,所以采用均值填充
    print(df['avg_price_top_last_12_valid_month'].describe())
    df['avg_price_top_last_12_valid_month'] = df['avg_price_top_last_12_valid_month'].fillna(df['avg_price_top_last_12_valid_month'].mean())
    #print(df['avg_price_top_last_12_valid_month'].describe())
    # 这儿再次查看结果,发现填充后的数据和填充前的数据，在统计上面没有发生太大变化
    
    # 分析下一组数据 空缺数在297到424之间
    cols = list(df_miss[df_miss.values<425].index)
    # 移除两个日期
    cols.remove('loans_latest_time')
    cols.remove('latest_query_time')
    # 发现这些数据基本都和违约时间等有联系 基本上偏差也不太大，采用众数填充
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    
    #日期类型采用后项填充
    df['loans_latest_time'] = df['loans_latest_time'].fillna(method='bfill')
    df['latest_query_time'] = df['latest_query_time'].fillna(method='bfill')
    print('\n填充日期:\n',df['loans_latest_time'].head(5))    
    # student_feature 学生特征
    print(df['student_feature'].value_counts())
    # 这列缺失过多，做删除处理
    df = df.drop('student_feature',axis=1)
    # 返回处理后的数据
    return df   

def train_test_spilt_local(df,test_size=0.3,random_state=2018):
    """切分数据集"""
    X = df.drop('status',axis=1)
    y = df['status']
    X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.3,random_state=2018)
    print(X_train.shape,X_test.shape)
    return X_train,y_train,X_test,y_test
    
    
def main():
    """第一节：EDA 探索性数据分析"""
    """1.1:读取和预览数据"""
    df = read_preview_data()
    """1.2:无关特征删除"""
    df = feature_analysis_clear(df)
    
    """1.2.1 对特征的相关性进行分析"""
    df = feature_corr_analysis(df)
    
    """1.3:数据类型转换"""
    df = transform_datatype(df)
    """
     对结果有价值的列: [trans_fail_top_count_enum_last_1_month', '
     history_fail_fee', 'apply_score', 'loans_score', 
     'max_cumulative_consume_later_1_month', 'latest_one_month_fail',
      'loans_overdue_count', 'trans_day_last_12_month',
       'repayment_capability', 'loans_undone_count',
        'consfin_avg_limit', 'trans_amount_3_month',
         'avg_price_last_12_month', 'trans_amount_increase_rate_lately',
          'latest_query_day', 'loans_latest_day', 'trans_fail_top_count_enum_last_12_month',
           'trans_days_interval', 'historical_trans_amount', 'trans_fail_top_count_enum_last_6_month', 
           'loans_consfin_max_limit', 'trans_activity_day', 'number_of_trans_from_2011', 
           'historical_trans_day', 'consfin_credit_limit', 'trans_days_interval_filter', 
           'apply_credibility', 'abs', 'history_suc_fee', 'latest_one_month_suc', 
           'loans_max_limit', 'first_transaction_day', 'loans_avg_limit', 'loans_consfin_avg_limit', 
           'loans_settle_count', 'loans_consfin_credit_limit', 'trans_top_time_last_6_month',
            'pawns_auctions_trusts_consume_last_6_month', 'loans_count', 'latest_six_month_apply',
             'pawns_auctions_trusts_consume_last_1_month', 'rank_trad_1_month', 
             'trans_consume_time_last_6_month', 'trans_activity_month']
    """
    """1.3.1: 提取对后期分析有用的列"""
    df = data_exploratory_analysis(df)
    
    
    """1.4:缺失值处理"""
#     df = missing_value_processing(df)
    """1.5: 数据切分"""
    X_train,y_train,X_test,y_test = train_test_spilt_local(df)
    """将数据写入文件"""
    df.to_csv('../dataset/data1.csv', index=None)
    
if __name__ == '__main__':
    main()