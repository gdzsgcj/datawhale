# -*- coding: utf-8 -*-
'''
Created on 2019年5月17日

@author: gdgzg
模型优化和调参

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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV,KFold


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



def model_adjustment_optimization_score(X_train,X_test,y_train,y_test):
    """
    模型调优和评价
    
    """
    metrics_score = []
    metrics_score.append(logistic_model_adjustment_optimization_score(X_train,X_test,y_train,y_test))
    metrics_score.append(svc_model_adjustment_optimization_score(X_train,X_test,y_train,y_test))
    metrics_score.append(tree_model_adjustment_optimization_score(X_train,X_test,y_train,y_test))
    metrics_score.append(random_forest_model_adjustment_optimization_score(X_train,X_test,y_train,y_test))
    metrics_score.append(xgboost_model_adjustment_optimization_score(X_train,X_test,y_train,y_test))
    print(metrics_score)

def logistic_model_adjustment_optimization_score(X_train,X_test,y_train,y_test):
    """
    逻辑回归模型优化调参 LogisiticRegression
    通过网格搜索最优参数
    """
    log_reg_params = {'penalty':['l1','l2'],'C':[1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3]}
    return model_adjustment_optimization(LogisticRegression(),'Logistic Regression',log_reg_params,X_train,X_test,y_train,y_test)

def svc_model_adjustment_optimization_score(X_train,X_test,y_train,y_test):
    """
    支持向量机 Support Vector Machine
    """
    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    return model_adjustment_optimization(SVC(),'Support Vector Machine',svc_params,X_train,X_test,y_train,y_test)
    
def tree_model_adjustment_optimization_score(X_train,X_test,y_train,y_test):   
    """
    决策树  Decision Tree
    """
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
    return model_adjustment_optimization(DecisionTreeClassifier(),'Decision Tree',tree_params,X_train,X_test,y_train,y_test) 

def random_forest_model_adjustment_optimization_score(X_train,X_test,y_train,y_test):
    """
    随机森林 RandomForestClassifier
    """
    rfc_params = {'n_estimators':range(10,100,10),'criterion':['gini','entropy'],'min_samples_leaf':[2, 4, 6],'max_features':['auto', 'sqrt', 'log2']}
    return model_adjustment_optimization(RandomForestClassifier(),'Random Forest', rfc_params,X_train,X_test,y_train,y_test)

def xgboost_model_adjustment_optimization_score(X_train,X_test,y_train,y_test): 
    """
    可伸缩灵活梯度提升 XGBoost
    """       
    xgb_params = {'max_depth':range(4,6,1), 'min_child_weight':range(4,12,2), 'gamma' : [i/10 for i in range(0,5)], 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]}
    return model_adjustment_optimization(XGBClassifier(), 'XGBoost', xgb_params, X_train, X_test, y_train, y_test)
    
def model_adjustment_optimization(model,model_name,param_grid,X_train,X_test,y_train,y_test):
    """
    训练模型
    获取最优模型
    """
    kfold = KFold(n_splits=5)
    grid_svc = GridSearchCV(model, param_grid = param_grid, cv=kfold, n_jobs=-1, scoring='roc_auc')
    grid_svc.fit(X_train, y_train)
    svc_best_estimator = grid_svc.best_estimator_
    print(grid_svc.best_params_)
    
    # 通过最优模型来训练
    ms = model_score_auc_curve(svc_best_estimator, model_name, X_test, y_test)
    return ms    

    
def model_score_auc_curve(model,model_name,X_test,y_test):    
    """
    用于输出模型评分
    和
    绘制ROC曲线
    input:
        model:模型
        model_name:模型名称
        X_test:
        y_test:
    output: 返回各项评分指标
        accuracy:
        precision:
        f1:
        recall:
        auc:
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_pred)
    
    print('{} 精确度 (accuracy):{:.2f}'.format(model_name,accuracy))
    print('{} 准确度(precision):{:.2f}'.format(model_name,precision))
    print('{} F1 Score :{:.2f}'.format(model_name,f1))
    print('{} 召回率(recall Score):{:.2f}'.format(model_name,recall))
    print('{} auc Score:{:.2f}'.format(model_name,auc))
    
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    plt.figure(figsize=(12,8))
    plt.title('{} 模型ROC曲线'.format(model_name), fontsize=18)
    plt.plot(fpr,tpr,label='{} 模型评分:{:.4f}'.format(model_name,roc_auc_score(y_test,y_pred)))    
    plt.plot([0,1],[0,1],'k--')
    plt.axis([-0.01,1,0,1])
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)',xy=(0.5,0.5),
            xytext=(0.6,0.3),arrowprops=dict(facecolor='#6E726D', shrink=0.05))
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='best')
    plt.show()
    ms = {}
    ms['model_name'] = model_name
    ms['accuracy'] = accuracy
    ms['precision'] = precision
    ms['f1'] = f1
    ms['recall'] = recall
    ms['auc'] = auc
    return ms
    

def main():
    # 1.1 加载切分后的数据集
    X_train,X_test,y_train,y_test = load_data()
    
    # 2.1 通过模型调参优化模型
    model_adjustment_optimization_score(X_train,X_test,y_train,y_test)
    
    

if __name__ == '__main__':
    main()