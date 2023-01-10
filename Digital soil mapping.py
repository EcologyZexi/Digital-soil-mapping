#!/usr/bin/env python
# coding: utf-8

# In[138]:


# suppress warning 
import warnings
warnings.filterwarnings('ignore')
import sys
#import gdal
from osgeo import gdal
from osgeo import gdalconst
from osgeo import gdal_array
from osgeo import osr
import rasterio
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
myfont = matplotlib.font_manager.FontProperties(fname='heiti.ttf',size=40)
plt.rcParams['axes.unicode_minus']=False
import matplotlib.cm as cm 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import KFold


# In[3]:


## Load rasters
import rasterio
covs = rasterio.open('elevation.tiff')


# In[88]:


# suppress warning 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
myfont = matplotlib.font_manager.FontProperties(fname='heiti.ttf',size=40)
plt.rcParams['axes.unicode_minus']=False


from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import time
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

import xgboost

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

from pandas.core.frame import DataFrame

from glob import glob
from pprint import pprint
#from shapely.geometry import Point
#from osgeo import gdal, ogr, gdal_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.plot import show, show_hist, reshape_as_raster, reshape_as_image
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV


# In[96]:


df_sample = pd.read_csv('sample_soc_v6.csv',encoding='unicode_escape')
df_sample


# In[6]:


df_sample.skew()


# In[7]:


def status(x) : 
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
    x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
    x.std(),x.skew(),x.kurt()],index=['count','min','idxmin','25% quantile',
    'median','75% quantile','mean','max','idmax','mad','var','std','skew','kurt'])


# In[328]:


import math
df_sample["lg_SOC"] = df_sample["SOC_0_5_cm"].apply(np.log)
df_sample


# In[333]:


df_sample['lg_SOC'].min()


# In[330]:


plt.figure(figsize=(12, 8))
plt.hist(df_sample['lg_SOC'],bins=100,facecolor='blue',edgecolor='black')
plt.xlabel("SOC")
plt.ylabel("frequency")
plt.xlim(0,10)
#x_ticks=np.arange(-50,50,2)
#plt.xticks(x_ticks)

#x_ticks=np.arange(2)
#plt.xticks(x_ticks)
plt.show()


# In[380]:


plt.figure(figsize=(12, 8))
plt.hist(df_sample['SOC_0_5_cm'],bins=28,facecolor='lightblue',edgecolor='black')
plt.xlabel("SOC (g $\mathregular{kg^{-1}}$)",size=15,family = 'Times New Roman')
plt.ylabel("Frequency",size=15,family = 'Times New Roman')
plt.xlim(0,100)
#x_ticks=np.arange(-50,50,2)
#plt.xticks(x_ticks)

plt.tick_params(labelsize=5)
plt.xticks(np.arange(0, 100, step=20),size = 14,family = 'Times New Roman')
plt.yticks(np.arange(0, 110, step=10),size = 14,family = 'Times New Roman')

plt.show()


# In[9]:


from scipy.stats import kstest
test_stat=kstest(df_sample['SOC_0_5_cm'],'norm')
test_stat


# In[10]:


df=df_sample.describe()
# Specify the name of the excel file
file_name = 'STATA.xlsx'
  
# saving the excelsheet
df.to_excel(file_name)


# In[97]:


# replace inf/-inf with np.nan
df_sample.replace([np.inf, -np.inf], np.nan, inplace=True)
df_sample=df_sample.replace(np.nan,0)
df_sample


# In[391]:


# select all features

x1 = df_sample.iloc[:,5:13]
x1


# In[392]:


x1 = pd.DataFrame(x1, dtype=np.float64)
x1


# In[340]:


x=x1.apply(lambda x1: (x1 - np.min(x1)) / (np.max(x1) - np.min(x1)))
x


# In[393]:


x1= x1.drop('ls_b31', axis=1)
x1= x1.drop('ls_b41', axis=1)

x1


# In[394]:


x1= x1.drop('slope1', axis=1)
#x1= x1.drop('twi1', axis=1)


x1


# In[108]:


x1= x1.drop('label', axis=1)


# In[101]:


data_corr=pd.DataFrame()
data_corr=x1
data_corr['label']=label
dat_clay_df=data_corr
dat_corr = dat_clay_df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(dat_corr, cmap='Spectral', annot=True, fmt=".2f")
plt.show()


# In[196]:


label1=label.apply(lambda label: (label - np.min(label)) / (np.max(label) - np.min(label)))
label1


# In[397]:


label=df_sample['SOC_0_5_cm']
label=DataFrame(label)
label


# In[482]:


# split the data into train, test set respectively
x_train, x_test, y_train, y_test = train_test_split(x1,label,random_state=0, train_size=0.8,test_size=0.2)


# In[483]:


print(y_train)
print(x_train)
print(y_test)


# In[504]:


# Initializing the Random Forest Regression model with 500 decision trees for predicting clay content
model_rf = RandomForestRegressor(n_estimators = 300,max_features=4)

# Fitting the Random Forest Regression model to the data
model_rf.fit(x_train, y_train['SOC_0_5_cm'])

# Predicting the target values of the test set
y_pred = model_rf.predict(x_test)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test['SOC_0_5_cm'], y_pred))))
print("\nRMSE: ", rmse, 'g/kg')

# Calculate and print out the mean absolute error (mae)
errors = abs(y_pred - y_test['SOC_0_5_cm'])
print('Mean Absolute Error:', round(np.mean(errors), 2), 'g/kg.')

# Calculate and print R-squared
r2 = r2_score(y_test['SOC_0_5_cm'].values.ravel(), y_pred)
print('R-squared: ', round(r2, 2))

mse=mean_squared_error(y_test['SOC_0_5_cm'].values.ravel(), y_pred)
print(mse)

## Plotting the results
plt.scatter(y_pred, y_test['SOC_0_5_cm'])
plt.show()


# In[503]:


pd.DataFrame(y_test)


# In[489]:


# Specify the name of the excel file
file_name = 'y_test.csv'
  
# saving the excelsheet
y_test.to_csv(file_name)


# In[470]:


y_test.mean()


# In[437]:


type(y_pred)


# In[456]:


from scipy import optimize
x=y_test.values.ravel()
y=y_pred


x2=np.linspace(-2,70)
y2=x2

C= round(r2_score(x,y),4)
rmse=round(np.sqrt(mean_squared_error(x,y)),3)

def f_1(x,A,B):
    return A*x+B

A1,B1=optimize.curve_fit(f_1,x,y)[0]
y3=A1*x+B1

fig, ax= plt.subplots(figsize=(7,7),dpi=200)
dian=plt.scatter(x,y,edgecolors=None,c='k',s=16,marker='s')

ax.plot(x2,y2,color='k',linewidth=1.5,linestyle='--')
ax.plot(x,y3,color='r',linewidth=1.5,linestyle='-')

fontdict1={'size':17,'color':'k','family':'Times New Roman'}

ax.set_xlabel('Measured SOC g/kg ',size = 15,family = 'Times New Roman')
ax.set_ylabel('Predicted SOC g/kg ',size = 15,family = 'Times New Roman')

ax.grid(False)
ax.set_xlim((0,60))   #设置坐标轴范围
ax.set_ylim((0,60))
ax.set_xticks(np.arange(0,60,step=10))
ax.set_yticks(np.arange(0,60,step=10))

for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_color('k')
ax.tick_params(left=True,bottom=True,direction='in',labelsize=14)

ax.text(1,45,r'$y=$'+str(round(A1,3))+'$x$'+'+'+str(round(B1,3)),fontdict=fontdict1)
ax.text(1,40,r'$R^2$='+str(round(C,3)),fontdict=fontdict1)
ax.text(1,35,r'$RMSE$='+str(rmse),fontdict=fontdict1)

plt.style.use('seaborn-darkgrid')
# Estimate the 2D histogramn
nbins = 150
H, xedges, yedges = np.histogram2d(x, y, bins=nbins)


plt.show()


# In[490]:


df_gwr = pd.read_csv('cb_v1.csv',encoding='unicode_escape')
df_gwr


# In[491]:


y_pred1=pd.DataFrame()
y_pred1=df_gwr.iloc[:,0]
np_pred=np.array(y_pred1)

y_original=pd.DataFrame()
y_original=df_gwr.iloc[:,1]
np_original=np.array(y_original)
np_original

np_original


# In[492]:



# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_original, y_pred1))))
print("\nRMSE: ", rmse, 'g/kg')



# Calculate and print R-squared
r2 = r2_score(y_original.values.ravel(), y_pred1)
print('R-squared: ', round(r2, 2))

mse=mean_squared_error(y_original.values.ravel(), y_pred1)
print(mse)

## Plotting the results
plt.scatter((y_original, y_pred1))
plt.show()


# In[507]:


from scipy import optimize
x=y_original.values.ravel()
y=y_pred1


x2=np.linspace(-2,120)
y2=x2

C= round(r2_score(x,y),4)
rmse=round(np.sqrt(mean_squared_error(x,y)),3)

def f_1(x,A,B):
    return A*x+B

A1,B1=optimize.curve_fit(f_1,x,y)[0]
y3=A1*x+B1

fig, ax= plt.subplots(figsize=(7,7),dpi=200)
dian=plt.scatter(x,y,edgecolors=None,c='k',s=16,marker='s')

ax.plot(x2,y2,color='k',linewidth=1.5,linestyle='--')
ax.plot(x,y3,color='r',linewidth=1.5,linestyle='-')

fontdict1={'size':17,'color':'k','family':'Times New Roman'}

ax.set_xlabel('Measured SOC g/kg ',size = 15,family = 'Times New Roman')
ax.set_ylabel('Predicted SOC g/kg ',size = 15,family = 'Times New Roman')

ax.grid(False)
ax.set_xlim((0,60))   #设置坐标轴范围
ax.set_ylim((0,60))
ax.set_xticks(np.arange(0,60,step=10))
ax.set_yticks(np.arange(0,60,step=10))

for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_color('k')
ax.tick_params(left=True,bottom=True,direction='in',labelsize=14)

ax.text(1,45,r'$y=$'+str(round(A1,3))+'$x$'+'+'+str(round(B1,3)),fontdict=fontdict1)
ax.text(1,40,r'$R^2$='+str(round(C,3)),fontdict=fontdict1)
ax.text(1,35,r'$RMSE$='+str(rmse),fontdict=fontdict1)

plt.style.use('seaborn-darkgrid')
# Estimate the 2D histogramn
nbins = 150
H, xedges, yedges = np.histogram2d(x, y, bins=nbins)




plt.show()


# In[ ]:


def regression_method1(model):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    
    pred_train=model.predict(x_train)
    ResidualSquare = (result - y_test)**2     #计算残差平方
    RSS = sum(ResidualSquare)   #计算残差平方和
    MSE = np.mean(ResidualSquare)       #计算均方差
    num_regress = len(result)   #回归样本个数
    print(f'n={num_regress}')
    print(f'R^2={score}')
    print(f'MSE={MSE}')
    print(f'RSS={RSS}')
    

    #MSE1 = np.square(np.subtract(y_test,result)).mean() 
    RMSE = math.sqrt(MSE)
    #print(f'MSE={MSE1}') 
    print(f'RMSE={RMSE}')
    print('回归截距: w0={}'.format(model.intercept_))  # w0: 截距
    print('回归系数: w1={}'.format(model.coef_))  # w1,..wm: 回归系数
    y_intercept=model.intercept_
    y_coef=model.coef_
    
    print("****************************训练")
    score1 = model.score(x_train,y_train)
    ResidualSquare1 = (pred_train - y_train)**2     #计算残差平方
    #RSS = sum(ResidualSquare)   #计算残差平方和
    MSE1 = np.mean(ResidualSquare1)       #计算均方差
    #num_regress = len(result)   #回归样本个数
    #print(f'n={num_regress}')
  #  print(f'R^211={model.r2_score}')
    print(f'R^2={score1}')
    print(f'MSE={MSE1}')
    #print(f'RSS={RSS}')
    print('回归截距: w0={}'.format(model.intercept_))  # w0: 截距
    print('回归系数: w1={}'.format(model.coef_))  # w1,..wm: 回归系数
    RMSE1 = math.sqrt(MSE1)
    print(f'RMSE={RMSE1}')
    
############绘制折线图##########
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('Difference of GLM Regression')
    plt.xlabel("Field survey test set")
    plt.ylabel("Measured DBH")
    plt.legend()        # 将样例显示出来
    plt.show()
    return result,y_intercept,y_coef,pred_train


# In[ ]:


b=pd.DataFrame()


# In[ ]:


import_level = model_rf.feature_importances_ #这个方法可以调取关于特征重要程度
a=[]
x_columns = x1.columns[0:]
#index = np.argsort(import_level)[::-1]
index = np.argsort(import_level)[::-1]
#        print(len(index))
for f in range(len(index)):
#    print(index[f])
    print("%2d %-*s %f" % (f + 1, 30, x_columns[index[f]], import_level[index[f]]))
    print(x_columns[index[f]])
    b['VI']=x_columns[index[f]]
    b['VIF']=import_level[index[f]]
    print(b)
    if import_level[index[f]]==import_level.min():
        a.append(index[f])
        #print(a)


# In[ ]:



import seaborn as sns

plt.figure(figsize=(6,12))

# 设定某一组为红色，其他组为蓝色
#my_pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
#sns.boxplot( x=df["species"], y=df["sepal_length"], palette=my_pal);


sns.boxplot(data=a,orient="h",color='skyblue', linewidth=1.2,notch=True, width=0.8)

plt.xlabel("Feature Importance",size=15,family = 'Times New Roman')
plt.ylabel("Predictor Variables",size=15,family = 'Times New Roman')

plt.xticks(size = 12,family = 'Times New Roman')
plt.yticks(size = 12,family = 'Times New Roman')

# plt.show()
plt.savefig('变量重要性.png',bbox_inches = 'tight',dpi=600)


# In[119]:


a=[]
for i in range(0,5000,100): # 迭代建立包含0-200棵决策树的RF模型进行对比
    rfc = RandomForestRegressor(n_estimators=i+1,random_state=1,n_jobs=-1)
    score = cross_val_score(rfc,x_train,y_train,cv=10,scoring='neg_mean_squared_error').mean()
    print(i)
    print("score:",score)
    a.append(score)
    print(a)


# In[121]:


ntree=pd.DataFrame(a)
ntree


# In[129]:


fig, ax = plt.subplots(figsize = (12,5))
#y1=trees.iloc[:,1]
#y2=trees.iloc[:,2]
y1=ntree.iloc[:,0]
#ax.fill_between(range(1,1001,20),y1, color ='lightblue',alpha=.5, linewidth=0)
#plt.plot(range(1,200,5),y3,color='blue',alpha=0.7,marker='o',markersize='4')
plt.plot(range(1,5000,100),y1,color='darkblue',alpha=0.7)
#plt.legend()
plt.xlabel("Number of trees selected",size=15,family = 'Times New Roman')
plt.ylabel("Accuracy",size=15,family = 'Times New Roman')

plt.tick_params(labelsize=5)
plt.xticks(np.arange(0, 5000, step=500),size = 12,family = 'Times New Roman')
plt.yticks(np.arange(-80, -60, step=2),size = 12,family = 'Times New Roman')


# In[136]:



from sklearn.model_selection import KFold


# In[145]:


TGMaxAcc=0

TGMeanAcc=[]
TGTLMaxAcc=[]#每次删除最不重要的变量里交叉验证的最大精度
TGTLMinAcc=[]
FGSort=pd.DataFrame()

df_TLAcc_all=pd.DataFrame()
S=pd.DataFrame()
S=x_train#特征变量个数
S = pd.DataFrame(S, dtype=np.float64)
#print(S)
label=y_train
label=label.values.tolist()
x_columns = S.columns[0:]
x_columns=x_columns.values.tolist()
x_columns
model=[]

unimportant_ftname=[]
unimportant_ftname_all=[]


for i in range(0,8,1):
    # split the data into train, test set respectively
    print("变量个数:",i)
#    print(i)
#    a=0
    FSort=pd.DataFrame()
    S=S.values.tolist()
    

#    print(train_data1)
#    print(train_label1)
    TLMaxAcc=0
    TLMeanAcc=0
    TLAcc=[]

    s=KFold(n_splits=10)
  #  print('s',s)
    j=-1
    for train,valid in s.split(S):
        j=j+1
        print(j,"折")
      #  print(train,valid)
        
        train_set=[]
        train_set_label=[]
        for p in range(len(train)):
          #  print(train[p])   
            valid_set=[]
            valid_set_label=[]       

            train_index=train[p]
            train_set.append(S[train_index])
            train_set_label.append(label[train_index])

        for q in range(len(valid)):
            valid_index=valid[q]
            valid_set.append(S[valid_index])
            valid_set_label.append(label[valid_index])            
        
        df_train_set = pd.DataFrame(train_set) 
        df_valid_set = pd.DataFrame(valid_set)       


        
        rfr1 = rfr(n_estimators=300,random_state=0)
        rfr1.fit(train_set, train_set_label)
        valid_pred1 = rfr1.predict(valid_set)
      #  print(train_set_label)
       # TLAcc1=mean_squared_error(valid_set_label, valid_pred1)*-1
        TLAcc1=r2_score(valid_set_label, valid_pred1)
        TLAcc.append(TLAcc1)
        print("TLAcc:",TLAcc)
        
        if j==9: #最后一折输出平均评分N折-1
            np_TLAcc=np.array(TLAcc)
            TLMeanAcc=np_TLAcc.mean()
            TGMeanAcc.append(TLMeanAcc)
            
            TGTLMaxAcc.append(TLMaxAcc)
            TLMinAcc=np_TLAcc.min()
            TGTLMinAcc.append(TLMinAcc)
            
            df_TLAcc=pd.DataFrame(TLAcc)
            df_TLAcc_all[i]=df_TLAcc
        
        if TLMaxAcc<=TLAcc[j]:
            TLMaxAcc=TLAcc[j]
            print("TLMaxAcc",TLMaxAcc)
          
            import_level = rfr1.feature_importances_ #这个方法可以调取关于特征重要程度
        #    x_columns = train_data1.columns[0:]
  
            index = np.argsort(import_level)[::-1]
            
            FSort=import_level
           # print("FSORT:",FSort)

            for f in range(len(index)):
                        #    print(index[f])
                #print("%2d %-*s %f" % (f + 1, 30, x_columns[index[f]], import_level[index[f]]))
                          #  print("%2d %-*s %f" % (f + 1,30, index[f], import_level[index[f]]))
            
                if import_level[index[f]]==import_level.min():
                    idx_feature=index[f]
                    print("最不重要的变量名序号:",idx_feature)
                    print("折最不重要的变量名:",x_columns[idx_feature])
                                             
    unimportant_ftname.append(x_columns[idx_feature])
            
    print('unimportant_ftname:',unimportant_ftname)
            
    if TGMaxAcc<=TLMaxAcc:
        TGMaxAcc=TLMaxAcc
        print("TLMaxAcc:",TLMaxAcc)
        FGSort=FSort
        unimportant_ftname_all=unimportant_ftname
        print('最优变量个数对应的丢失特征:',unimportant_ftname_all)
        print('最优变量个数对应的丢失特征个数:',len(unimportant_ftname_all))


    print("删除变量时最不重要的变量名:",x_columns[idx_feature])
    del x_columns[idx_feature]
    S=pd.DataFrame(S)
    S=S.drop(S.columns[idx_feature],axis = 1)
    
    print('***************************************************************************')
    
print("TGMaxAcc:",TGMaxAcc)
print("FGSORT:",FGSort)
print("TGTLMeanAcc:",TGMeanAcc)#最后的平均精确度
print("TGTLMaxAcc:",TGTLMaxAcc) 
print("TGTLMinAcc:",TGTLMinAcc) 
df_TLAcc_all


# In[257]:


TGMaxAcc=0

TGMeanAcc=[]
TGTLMaxAcc=[]#每次删除最不重要的变量里交叉验证的最大精度
TGTLMinAcc=[]
npFsort=[]
FGSort=pd.DataFrame()

df_TLAcc_all=pd.DataFrame()
S=pd.DataFrame()
S=x_train#特征变量个数
S = pd.DataFrame(S, dtype=np.float64)
#print(S)
label=y_train
label=label.values.tolist()
x_columns = S.columns[0:]
x_columns=x_columns.values.tolist()
x_columns
model=[]
S=S.values.tolist()
unimportant_ftname=[]
unimportant_ftname_all=[]

FSort1=pd.DataFrame()


#    a=0
FSort=pd.DataFrame()



#    print(train_data1)
#    print(train_label1)
TLMaxAcc=0
TLMeanAcc=0
TLAcc=[]
df_ip=pd.DataFrame()
s=KFold(n_splits=10)
#  print('s',s)

for i in range(0,10,1):
    j=-1
    for train,valid in s.split(S):
        j=j+1
        print(j,"折")
      #  print(train,valid)

        train_set=[]
        train_set_label=[]
        for p in range(len(train)):
          #  print(train[p])   
            valid_set=[]
            valid_set_label=[]       

            train_index=train[p]
            train_set.append(S[train_index])
            train_set_label.append(label[train_index])

        for q in range(len(valid)):
            valid_index=valid[q]
            valid_set.append(S[valid_index])
            valid_set_label.append(label[valid_index])            

        df_train_set = pd.DataFrame(train_set) 
        df_valid_set = pd.DataFrame(valid_set)       



        rfr1 = rfr(n_estimators=300,random_state=0)
        rfr1.fit(train_set, train_set_label)
        valid_pred1 = rfr1.predict(valid_set)
      #  print(train_set_label)
       # TLAcc1=mean_squared_error(valid_set_label, valid_pred1)*-1
        TLAcc1=r2_score(valid_set_label, valid_pred1)
        TLAcc.append(TLAcc1)
        print("TLAcc:",TLAcc)

        if j==9: #最后一折输出平均评分N折-1
            np_TLAcc=np.array(TLAcc)
            TLMeanAcc=np_TLAcc.mean()
            TGMeanAcc.append(TLMeanAcc)

            TGTLMaxAcc.append(TLMaxAcc)
            TLMinAcc=np_TLAcc.min()
            TGTLMinAcc.append(TLMinAcc)

            df_TLAcc=pd.DataFrame(TLAcc)
            df_TLAcc_all[i]=df_TLAcc



        import_level = rfr1.feature_importances_ #这个方法可以调取关于特征重要程度
    #    x_columns = train_data1.columns[0:]

        df_ip=pd.DataFrame(import_level)
        #np_df_ip=np.array(import_level)
        #npFsort.append(np_df_ip)
        
        FSort1[j]=pd.DataFrame(df_ip)

        print('ddd',df_ip)

print('***************************************************************************')

print("TGMaxAcc:",TGMaxAcc)
print("FGSORT:",FGSort)
print("TGTLMeanAcc:",TGMeanAcc)#最后的平均精确度
print("TGTLMaxAcc:",TGTLMaxAcc) 
print("TGTLMinAcc:",TGTLMinAcc) 
df_TLAcc_all


# In[300]:


print(FSort1)


# In[311]:


ip = pd.DataFrame()
ip=FSort1
ip.sort_values(0,ascending=False)
ip1=ip
ip1


# In[301]:


ip = pd.DataFrame()
ip2=ip1.transpose()
ip2


# In[381]:


names=['slope', 'ls_b4','aspect', 'ls_b3','NDVI','twi','radk','elevation']
ip2.columns=names
ip2


# In[382]:



import seaborn as sns

plt.figure(figsize=(6,12))

# 设定某一组为红色，其他组为蓝色
#my_pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
#sns.boxplot( x=df["species"], y=df["sepal_length"], palette=my_pal);


sns.boxplot(data=ip2,orient="h",color='skyblue', linewidth=1,notch=True, width=0.4)

plt.xlabel("Feature Importance",size=15,family = 'Times New Roman')
plt.ylabel("Predictor Variables",size=15,family = 'Times New Roman')

plt.xticks(size = 15,family = 'Times New Roman')
plt.yticks(size = 15,family = 'Times New Roman')

# plt.show()
plt.savefig('variable importance.png',bbox_inches = 'tight',dpi=600)


# ## Prediction model

# In[284]:



def loadTiff(in_image):
    im = gdal.Open(in_image)
    print(type(im))
    Col = im.RasterXSize
    Row = im.RasterYSize

    imarray = np.array(im.ReadAsArray())
    imarray2d = imarray.transpose(1,2,0).reshape(-1,4)#2个变量填2
    return im,imarray,imarray2d,Col,Row,


# In[285]:


fp = ('C:/Users/zexir/Desktop/DSM test/ML_spatial/bcv2.tif')
im,imarray,imarray2d,Col,Row=loadTiff(fp)


# In[295]:


pred_array= model_rf.predict(imarray2d)


# In[296]:


pred_array


# In[297]:


pred_array2d = pred_array.reshape(Row,Col)
print(pred_array2d)


# In[298]:


def write_geotiff(filename, arr, im):
    arr_type = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")
    out_im = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_im.SetProjection(im.GetProjection())
    out_im.SetGeoTransform(im.GetGeoTransform())
    band = out_im.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


# In[299]:


filename=('C:/Users/zexir/Desktop/DSM test/ML_spatial/pred_v2.tif')
write_geotiff(filename,pred_array2d,im)

