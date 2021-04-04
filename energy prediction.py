#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

class Linear_Regression():
    def __init__(self):
        self.split = [2,3,4,5,10]
        
    def dataset_split(self):
        df = pd.read_csv(r"C:\Users\dell\Downloads/energydata_complete.csv" )
        X = df[['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','Press_mm_hg','RH_out','Windspeed']][:].values.reshape(19735, 21)
        y =  df[['rv1']][:].values.reshape(19735, 1)
        return X,y
    
    def model_creation(self, X, y):
        acc = []
        for j in self.split:
            kfold = model_selection.KFold(n_splits=j)
            model = linear_model.LinearRegression()
            results = model_selection.cross_val_score(model, X, y, cv=kfold)
            accuracy_score = results.mean()
            acc.append(accuracy_score)
        return acc
    
    def graph(self, acc_score):
        
        plt.plot(self.split, acc_score)
        plt.title("Linear Regression \n Variations in Accuracy \n with K-Fold Splits", pad=20)
        plt.ylabel("Accuracy", fontsize = "large")
        plt.xlabel('K-Fold Splits', fontsize = "large")
        plt.legend(["Linear Model"], loc='lower right')
        plt.show()

    
linear = Linear_Regression()
X,y = linear.dataset_split()
accuracy = linear.model_creation(X,y)
print("Accuracy wrt K-fold", accuracy)

linear.graph(accuracy)


# In[ ]:




