#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[2]:


data=pd.read_csv('cardio_train.csv',sep=';')
data.head()


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.drop('id',axis=1,inplace=True)

data.drop_duplicates(inplace=True)


# In[6]:


data.shape


# In[7]:


plt.figure(figsize=(20,18))
plotnumber=1

for column in data[['age','height','weight','ap_hi','ap_lo']]:
    if plotnumber<6:
        ax=plt.subplot(2,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=10)
        
        plotnumber+=1
plt.show()


# In[8]:


from scipy.stats import zscore


# In[9]:


z_score=zscore(data[['age','height','weight','ap_hi','ap_lo']])

abs_z_score=np.abs(z_score)

filtering_entry=(abs_z_score<3).all(axis=1)

data=data[filtering_entry]

data.describe()


# In[10]:


plt.figure(figsize=(20,18))
plotnumber=1

for column in data[['age','height','weight','ap_hi','ap_lo']]:
    if plotnumber<6:
        ax=plt.subplot(2,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=10)
        
        plotnumber+=1
plt.show()


# In[11]:


data_corr=data.corr().abs()


# In[12]:


plt.figure(figsize=(18,14))
sns.heatmap(data_corr,annot=True,annot_kws={'size':10})
plt.show()


# In[13]:


x=data.drop(['cardio'],axis=1)
y=data['cardio']


# In[14]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[15]:


x_scaled.shape[1]


# In[16]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[17]:


vif=pd.DataFrame()

vif['vif']=[variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]

vif['features']=x.columns

vif


# In[18]:


data.shape


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state=6999)


# In[20]:


def metric_score(clf,x_train,x_test,y_train,y_test,train=True):
    if train:
        y_pred=clf.predict(x_train)
        print("accuracy score of training score : ",accuracy_score(y_train,y_pred))
        
    elif train==False:
        pred=clf.predict(x_test)
        print('accuracy score of testing score : ',accuracy_score(y_test,pred))
        print('\n \n classification report \n \n : ',classification_report(y_test,pred))


# In[21]:


random_clf=RandomForestClassifier()
random_clf.fit(x_train,y_train)


# In[22]:


metric_score(random_clf,x_train,x_test,y_train,y_test,train=True)

metric_score(random_clf,x_train,x_test,y_train,y_test,train=False)


# In[23]:


param_grid={'n_estimators':[13,15],
            'criterion':['entropy','gini'],
           'max_depth':[10,15],
           'min_samples_split':[10,11],
            'min_samples_leaf':[5,6]
           }


# In[24]:


gridsearch=GridSearchCV(estimator=random_clf,param_grid=param_grid)


# In[25]:


gridsearch.fit(x_train,y_train)


# In[26]:


gridsearch.best_params_


# In[27]:


random_clf=gridsearch.best_estimator_
random_clf.fit(x_train,y_train)


# In[28]:


metric_score(random_clf,x_train,x_test,y_train,y_test,train=True)

metric_score(random_clf,x_train,x_test,y_train,y_test,train=False)


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import plot_roc_curve


# In[30]:


lr=LogisticRegression()
dt=DecisionTreeClassifier()
kn=KNeighborsClassifier()
random_clf=RandomForestClassifier()


# In[31]:


x=data.drop(['cardio'],axis=1)
y=data['cardio']


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=98)


# In[33]:


scaled=StandardScaler()
x_scaled=scaled.fit(x_train,y_train)


# In[34]:


lr.fit(x_train,y_train)


# In[35]:


metric_score(lr,x_train,x_test,y_train,y_test,train=True)
metric_score(lr,x_train,x_test,y_train,y_test,train=False)


# In[36]:


param_grid={'penalty':['l1','l2'],
           'C':np.logspace(-4,4,50)}


# In[37]:


gridsearch=GridSearchCV(estimator=lr,param_grid=param_grid)


# In[38]:


gridsearch.fit(x_train,y_train)


# In[39]:


gridsearch.best_params_


# In[40]:


lr=gridsearch.best_estimator_


# In[41]:


lr.fit(x_train,y_train)


# In[42]:


metric_score(lr,x_train,x_test,y_train,y_test,train=True)
metric_score(lr,x_train,x_test,y_train,y_test,train=False)


# In[43]:


dt.fit(x_train,y_train)


# In[44]:


metric_score(dt,x_train,x_test,y_train,y_test,train=True)
metric_score(dt,x_train,x_test,y_train,y_test,train=False)


# In[45]:


params={'criterion':['gini','entropy'],
       'max_depth':range(10,15),
       'min_samples_leaf':range(2,10),
       'min_samples_split':range(3,10),
       'max_leaf_nodes':range(5,10)}


# In[46]:


grd=GridSearchCV(estimator=dt,param_grid=params,cv=5,n_jobs=-1)


# In[47]:


grd.fit(x_train,y_train)


# In[48]:


grd.best_params_


# In[49]:


dt=grd.best_estimator_


# In[50]:


dt.fit(x_train,y_train)


# In[51]:


metric_score(dt,x_train,x_test,y_train,y_test,train=True)
metric_score(dt,x_train,x_test,y_train,y_test,train=False)


# In[52]:


kn=KNeighborsClassifier()


# In[53]:


kn.fit(x_train,y_train)


# In[54]:


metric_score(kn,x_train,x_test,y_train,y_test,train=True)
metric_score(kn,x_train,x_test,y_train,y_test,train=False)


# In[55]:


param={'algorithm':['kd_tree','brute'],
      'leaf_size':[3,5,6,7,8,9,10],
      'n_neighbors':[3,5,7,9,11,13]}


# In[56]:


grd_search=GridSearchCV(estimator=kn,param_grid=param)


# In[57]:


grd_search.fit(x_train,y_train)


# In[58]:


grd_search.best_params_


# In[59]:


kn=grd_search.best_estimator_


# In[60]:


kn.fit(x_train,y_train)


# In[61]:


metric_score(kn,x_train,x_test,y_train,y_test,train=True)
metric_score(kn,x_train,x_test,y_train,y_test,train=False)


# In[62]:


disp=plot_roc_curve(dt,x_train,y_train)

plot_roc_curve(lr,x_train,y_train,ax=disp.ax_)

plot_roc_curve(kn,x_train,y_train,ax=disp.ax_)

plot_roc_curve(random_clf,x_train,y_train,ax=disp.ax_)

plt.legend(prop={'size':10},loc='lower right')

plt.show()


# In[63]:


disp=plot_roc_curve(dt,x_test,y_test)

plot_roc_curve(lr,x_test,y_test,ax=disp.ax_)

plot_roc_curve(kn,x_test,y_test,ax=disp.ax_)

plot_roc_curve(random_clf,x_test,y_test,ax=disp.ax_)

plt.legend(prop={'size':10},loc='lower right')

plt.show()


# In[ ]:




