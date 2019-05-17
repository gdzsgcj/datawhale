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
    
    for model_name,model in classifiers.items():
        model.fit(X_train,y_train)
#         cross_val_score_local = cross_val_score(model,X_train,y_train,cv=5)
        print('*' * 10,'模型名称:',model_name,'*' * 10)
#         print(',交叉验证得分:',round(cross_val_score_local.mean() * 100,2),'%')
        y_pred = model.predict(X_test)
        print('Accuracy Score:{:.2f}'.format(accuracy_score(y_test,y_pred)))
        print('Precision Score:{:.2f}'.format(precision_score(y_test,y_pred)))
        print('F1 Score:{:.2f}'.format(f1_score(y_test,y_pred)))
        print('Recall Score:{:.2f}'.format(recall_score(y_test,y_pred)))
        # 绘制ROC曲线
        fpr,tpr,threshold = roc_curve(y_test,y_pred)
        plt.figure(figsize=(10,8))
        plt.title('{} 模型 ROC曲线'.format(model_name),fontsize=18)
        plt.plot(fpr,tpr,label='{} 模型 得分:{:.4f}'.format(model_name,roc_auc_score(y_test,y_pred)))
        plt.plot([0,1],[0,1],'k--')
        plt.axis([-0.01,1,0,1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc='best')
        plt.show()
        
        
def main():
    # 1.1 加载切分后的数据集
    X_train,X_test,y_train,y_test = load_data()
    
    # 逻辑回归预测 logisticRegression
    model_predict(X_train,X_test,y_train,y_test)
    

if __name__ == '__main__':
    main()

