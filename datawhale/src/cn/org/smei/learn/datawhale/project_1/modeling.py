# -*- coding: utf-8 -*-
'''
Created on 2019年5月15日

@author: gdgzg
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.ranking import roc_curve
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10,6
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score,precision_score,f1_score,\
            recall_score,precision_recall_curve,roc_auc_score,roc_curve
from sklearn.model_selection import learning_curve,train_test_split,cross_val_score

DATA_FILE = '../dataset/data2.csv' 

pd.set_option('display.max_columns',None)

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
    print(df.shape)
    X = df.drop('status',axis=1)
    y = df['status']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2018)
    return X_train,X_test,y_train,y_test
    
def model_predict(X_train,X_test,y_train,y_test):
    """
    采用各类模型进行预测
    """
    classifiers = {
        'LogisticRegression' : LogisticRegression(C=0.001),
        'Support Vector Machine Classifier' : SVC(),
        'Decision Tree Classifier' : DecisionTreeClassifier(),
        'Random Forest Classifier' : RandomForestClassifier(),
        'Xgboost Classifier' : XGBClassifier()
        }
    
    model_metrics = []
    
    for model_name,model in classifiers.items():
        model.fit(X_train,y_train)
#         cross_val_score_local = cross_val_score(model,X_train,y_train,cv=5)
        print('*' * 10,'模型名称:',model_name,'*' * 10)
#         print(',交叉验证得分:',round(cross_val_score_local.mean() * 100,2),'%')

        model_sc = {}
        
        """这是训练集上的评价"""
        y_pred_t = model.predict(X_train)
        accuracy_t = accuracy_score(y_train,y_pred_t)
        precision_t = precision_score(y_train,y_pred_t)
        f1_t = f1_score(y_train,y_pred_t)
        recall_t = recall_score(y_train,y_pred_t)
        auc_t = roc_auc_score(y_train,y_pred_t)        
        
        model_sc['model_name'] = model_name
        model_sc['model_data_sort'] = 'Train Data'
        model_sc['accuracy'] = accuracy_t
        model_sc['precision'] = precision_t
        model_sc['f1'] = f1_t
        model_sc['recall'] = recall_t
        model_sc['auc'] = auc_t
        model_metrics.append(model_sc)
        
        print('\n训练集：Accuracy Score:{:.2f}'.format(accuracy_t))
        print('\n训练集：Precision Score:{:.2f}'.format(precision_t))
        print('\n训练集：F1 Score:{:.2f}'.format(f1_t))
        print('\n训练集：Recall Score:{:.2f}'.format(recall_t))
        print('\n训练集：Auc Roc Score:{:.2f}'.format(auc_t))

        """这是才测试集上的评价"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)   
         
        model_sc = {}
        model_sc['model_name'] = model_name
        model_sc['model_data_sort'] = 'Test Data'
        model_sc['accuracy'] = accuracy
        model_sc['precision'] = precision
        model_sc['f1'] = f1
        model_sc['recall'] = recall
        model_sc['auc'] = auc
        model_metrics.append(model_sc)            
        
        print('\n测试集：Accuracy Score:{:.2f}'.format(accuracy))
        print('\n测试集：Precision Score:{:.2f}'.format(precision))
        print('\n测试集：F1 Score:{:.2f}'.format(f1))
        print('\n测试集：Recall Score:{:.2f}'.format(recall))
        print('\n测试集：Auc Roc Score:{:.2f}'.format(auc))
        # 绘制ROC曲线
        fpr,tpr,threshold = roc_curve(y_test,y_pred)
        fpr_t,tpr_t,threshold_t = roc_curve(y_train,y_pred_t)
        plt.figure(figsize=(10,8))
        plt.title('{} 模型 ROC曲线'.format(model_name),fontsize=18)
        plt.plot(fpr_t,tpr_t, label='{} 模型 在训练数据 得分:{:.4f}'.format(model_name,auc_t))
        plt.plot(fpr,tpr,label='{} 模型 在测试数据 得分:{:.4f}'.format(model_name,auc))
        plt.plot([0,1],[0,1],'k--')
        plt.axis([-0.01,1,0,1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc='best')
        plt.show()
        
    return model_metrics
        
        
def main():
    # 1.1 加载切分后的数据集
    X_train,X_test,y_train,y_test = load_data()
    
    # 多种模型预测
    model_metrics = model_predict(X_train,X_test,y_train,y_test)
    model_metrics_df = pd.DataFrame(model_metrics)
    model_metrics_df.set_index(['model_name','model_data_sort'], inplace=True)
    
    print(model_metrics_df)

if __name__ == '__main__':
    main()

