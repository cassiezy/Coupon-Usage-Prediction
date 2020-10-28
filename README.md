# Coupon Usage Prediction
### Python sklearn: Find out whom to send the coupon by predicting whether customers will use the coupon next month.

### Import packages
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
```

### Import data
```Python
pdd=pd.read_csv('pinduoduo.csv')
pdd.head()
```

### EDA
```
pdd.info()
```
##### No missing value found!
```
pdd.describe(include='all')
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/description.png)

##### *Notice:* 
1. abnormal value for age :95
2. distribution of coupon_used_in_last6_monthis discrete，try binning this variable
```
sum(pdd.duplicated())
```
##### No duplicates!

#### Target Variable
```
pdd=pdd.rename(columns={'coupon_ind':'flag'})
```
```
pdd.flag.value_counts(1)
```
|Flag |percentage|
|-----|--------|
|0    |0.883043|
|1    |0.116957|
##### 11.6% of customers used coupons before - acceptable!

### Data Preparation
#### Numeric Variable
1. age
```
pdd['age'].plot(kind='hist', bins=40)
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/1.png)

##### *Notice: majority are aged 25-60, try binning them later* 

```
# Find out the outliers
diff=pdd['age'].describe()['75%']-pdd['age'].describe()['25%']
max=pdd['age'].describe()['75%']+1.5*diff
max
```
70.5

```
# will lose 1.12% of data if filter out age > 70 - accetable
(pdd.shape[0]-pdd[pdd['age']<=70].shape[0])/pdd.shape[0]
```
0.01121775881818541
```
# eliminate the outliers
pdd=pdd[pdd['age']<=70]
```

2. coupon_used_in_last6_month
```
pdd['coupon_used_in_last6_month'].describe()
pdd['coupon_used_in_last6_month'].plot(kind='hist', bins=40);
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/2.png)

##### *Notice: long tail, need to find out the outliers* 
```
diff=pdd['coupon_used_in_last6_month'].describe()['75%']-pdd['coupon_used_in_last6_month'].describe()['25%']
max=pdd['coupon_used_in_last6_month'].describe()['75%']+1.5*diff
max
```
6.0
```
# will lose 6.78% of data if filter out coupon_used_in_last6_month > 6 - accetable
(pdd.shape[0]-pdd[pdd['coupon_used_in_last6_month']<=6].shape[0])/pdd.shape[0]
```
0.06822993648384133
```
# eliminate the outliers
pdd=pdd[pdd['coupon_used_in_last6_month']<=6]
```


3. coupon_used_in_last_month
```
pdd['coupon_used_in_last_month'].describe()
pdd[pdd['coupon_used_in_last_month']==0].shape[0]/pdd.shape[0]
```
0.8104180064308681
##### *80% of users didn't use coupons last month.*

#### Categorical Variable
1. job
```
pdd.job.value_counts()
```
|Job    | Count|
|----| ---|
|blue-collar     | 5082|
|management      | 4885|
|technician      | 3914|
|admin.          | 2726|
|services       |  2183|
|retired         |  990|
|self-employed    | 823|
|entrepreneur     | 787|
|unemployed       | 676|
|housemaid        | 606|
|student           |512|
|unknown           |141|

2. marital
```
pdd.marital.value_counts().plot(kind='pie')
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/3.png)

3. default
```
pdd.default.value_counts()
```
|default |percentage|
|-----|--------|
|no    |22922|
|yes    |403|

4. returned
```
pdd.returned.value_counts()
```
|default |percentage|
|-----|--------|
|no    |10184|
|yes    |13141|
```
sns.countplot(y='returned',hue='flag',data=pdd)
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/4.png)


5. loan
```
pdd.loan.value_counts()
```
|loan |percentage|
|-----|--------|
|no    |19576|
|yes    |3749|

```
pdd=pd.get_dummies(pdd)
pdd.head()
```
5 rows × 26 columns
```
pdd.drop(['ID','default_no','returned_no','loan_no'],axis=1,inplace=True)
```
5 rows × 22 columns

```
pdd.groupby(['flag']).mean()
```
|flag	|age	|coupon_used_in_last6_month	|coupon_used_in_last_month	|job_admin.	|job_blue-collar	|job_entrepreneur	|job_housemaid	|job_management	|job_retired	|job_self-employed|	...	|job_student	|job_technician	|job_unemployed	|job_unknown	|marital_divorced	|marital_married	|marital_single	|default_yes	|returned_yes	|loan_yes|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|																		
|0	|40.524198	|2.164091	|0.259767	|0.116618	|0.228669	|0.035131	|0.027017|	0.203110	|0.039213	|0.035180|	...	|0.017979|	0.167687	|0.027308	|0.005977	|0.115500|	0.608358	|0.276142	|0.018416|	0.589067|	0.169096|
|1	|40.301275|	1.888525|	0.540619	|0.118761	|0.136976	|0.023315|	0.018215|	0.256831|	0.066667|	0.036066	|...	|0.051730|	0.168670|	0.041530|	0.006557|	0.112568|	0.517304	|0.370128	|0.008743	|0.370856	|0.097996|

##### Users who meet the following criteria are more likely to use coupons:
+ More coupons used last month
+ Didn't used too many coupons in the past six months. (More usage in the past six months means more shopping, so it's less likely for them to continue shopping this time and less likely to use coupons.)
+ No returning behavior
+ Retired/students/unemployed
+ Single people
+ Non-credit card users



#### Correlation Analysis
```
pdd.corr()['flag'].sort_values(ascending=False)
```
|Variable                  |Correlation  |
|-------   |  -------|
|flag                        |  1.000000|
|coupon_used_in_last_month  |   0.124839|
|job_student                |   0.074227|
|marital_single              |  0.066935|
|job_retired                 |  0.043882|
|job_management              |  0.042543|
|job_unemployed             |   0.027318|
|job_unknown                 |  0.002414|
|job_admin.                 |   0.002150|
|job_self-employed          |   0.001547|
|job_technician              |  0.000848|
|marital_divorced           |  -0.002960|
|age                          |-0.007172|
|job_housemaid              |  -0.017829|
|job_entrepreneur             |-0.021087|
|job_services                | -0.023711|
|default_yes                  |-0.023920|
|marital_married              |-0.059833|
|loan_yes                     |-0.062380|
|coupon_used_in_last6_month  | -0.067464|
|job_blue-collar              |-0.071575|
|returned_yes                | -0.141774|
```
q1=['coupon_used_in_last_month', 'job_student', 'marital_single','job_retired','job_management','job_unemployed',
    'marital_married','loan_yes','coupon_used_in_last6_month','job_blue-collar','returned_yes']
sns.heatmap(pdd[q1].corr())
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/5.png)



## Model building
```
y=pdd['flag']
x=pdd[['coupon_used_in_last_month', 'job_student', 'marital_single','job_retired','job_management','job_unemployed',
    'marital_married','loan_yes','coupon_used_in_last6_month','job_blue-collar','returned_yes']]
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn import linear_model
lr=linear_model.LogisticRegression()

lr.fit(x_train,y_train)
```
```
lr.intercept_
```
array([-1.29772036])
```
lr.coef_
```
array([[ 0.50429086,  0.59938599,  0.2592328 ,  0.41205734,  0.21818936,
         0.33533269, -0.0722816 , -0.49721391, -0.26037011, -0.25374964,
        -0.85610703]])

```
y_pred_train=lr.predict(x_train)
y_pred_test=lr.predict(x_test)
```
#### Accuracy
```
import sklearn.metrics as metrics
```
##### Training set
```
metrics.confusion_matrix(y_train,y_pred_train)
metrics.accuracy_score(y_train,y_pred_train)
```
0.8807496784467447

##### Test set
```
metrics.confusion_matrix(y_test,y_pred_test)
metrics.accuracy_score(y_train,y_pred_train)
```
0.883252357816519



## Model optimization
### A. Exclude the outliers (already done)
### B. train_test_split:50%/50%
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1)
from sklearn import linear_model
lr=linear_model.LogisticRegression()
lr.fit(x_train,y_train)

y_pred_train=lr.predict(x_train)
y_pred_test=lr.predict(x_test)
```
```
metrics.accuracy_score(y_train,y_pred_train)
```
0.8800377293774653
```
metrics.accuracy_score(y_test,y_pred_test)
```
0.8829632170110606

##### Accuracy decreased, keep using 7/3 split.

### C. age binning
```
pdd['age'].plot(kind='hist', bins=40)
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/6.png)

```
bins=[0,30,40,50,60,70]
labels=['<30','30-40','40-50','50-60','60-70']#为组创建标签名称
pdd['age_group']=pd.cut(pdd.age, bins, right=False, labels=labels)
sns.countplot(y='age_group', hue='flag',data=pdd)
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/7.png)

```
y=pdd['flag']
x=pdd[['age_group_60-70','age_group_<30','coupon_used_in_last_month', 'job_student', 'marital_single','job_retired','job_management','job_unemployed',
    'marital_married','loan_yes','coupon_used_in_last6_month','job_blue-collar','returned_yes']]
```
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn import linear_model
lr=linear_model.LogisticRegression()
lr.fit(x_train,y_train)

y_pred_train=lr.predict(x_train)
y_pred_test=lr.predict(x_test)
```
```
metrics.accuracy_score(y_train,y_pred_train)
```
0.9008721749249709
```
metrics.accuracy_score(y_test,y_pred_test)
```
0.893252357816519

##### Accuracy increased.

## Business proposal
#### Send coupons to people below to increase sales:
1. students
2. single people younger than 30
3. retired people aged 60 to 70
4. unemployed people
5. people who used coupons last month

#### Avoid sending coupons to people below to reduce potential cost:
1. married blue-collar
2. users with a high return record
3. heavy coupon users in the past six months
4. users who prefer to pay by credit card











