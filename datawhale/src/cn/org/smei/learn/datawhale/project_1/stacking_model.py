# -*- coding: utf-8 -*-
'''
Created on 2019年5月22日

模型融合
@author: gdgzg
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

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,\
            recall_score,precision_recall_curve,roc_auc_score,roc_curve
from mlxtend.classifier import StackingClassifier

DATA_FILE = '../dataset/data2.csv' 


def load_data():
    """
    加载数据和分割数据
    return
     X_train,
     X_test,
     y_train,
     y_test
    """
    df = pd.read_csv(DATA_FILE)
#     print(df.shape)
#     print(df.info())
    X = df.drop('status',axis=1)
    y = df['status']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2018)
    return X_train,X_test,y_train,y_test

    
def model_processing(X_train,X_test,y_train,y_test):
    log_reg = LogisticRegression(C=0.01, penalty='l2')
    svc = SVC(C=0.7, kernel='linear')
    tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
    rf_clf = RandomForestClassifier(n_estimators=70,criterion='entropy', max_features='auto',min_samples_leaf=6)
    xgb = XGBClassifier(gamma=0.3, max_depth=4, min_child_weight=8,reg_alpha=0.05)
    
    sclf = StackingClassifier(classifiers=[log_reg,svc,tree_clf,rf_clf],meta_classifier=xgb)
    sclf.fit(X_train,y_train)
    y_pred_train = sclf.predict(X_train)
    y_pred = sclf.predict(X_test)
    
    print('*' * 30,'在训练集上的得分' )
    
    accuracy = accuracy_score(y_train,y_pred_train)
    precision = precision_score(y_train,y_pred_train)
    f1 = f1_score(y_train,y_pred_train)
    recall = recall_score(y_train,y_pred_train)
    auc = roc_auc_score(y_train,y_pred_train)
    model_name = '堆叠模型-训练集'
     
    print('{} 精确度 (accuracy):{:.2f}'.format(model_name,accuracy))
    print('{} 准确度(precision):{:.2f}'.format(model_name,precision))
    print('{} F1 Score :{:.2f}'.format(model_name,f1))
    print('{} 召回率(recall Score):{:.2f}'.format(model_name,recall))
    print('{} auc Score:{:.2f}'.format(model_name,auc))
    
    
    print('*' * 30,'在测试集上的得分' )
    
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_pred)
    model_name = '堆叠模型'
     
    print('{} 精确度 (accuracy):{:.2f}'.format(model_name,accuracy))
    print('{} 准确度(precision):{:.2f}'.format(model_name,precision))
    print('{} F1 Score :{:.2f}'.format(model_name,f1))
    print('{} 召回率(recall Score):{:.2f}'.format(model_name,recall))
    print('{} auc Score:{:.2f}'.format(model_name,auc))    
    
def main():
    # 1 加载数据源
    X_train,X_test,y_train,y_test = load_data()
    model_processing(X_train,X_test,y_train,y_test)
    

if __name__ == '__main__':
    main()
    
        
    
    
        