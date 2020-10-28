# Coupon Usage Prediction
### Python sklearn: Predict whether customers will use the coupo next month.

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
#### No missing value found!
```
pdd.describe(include='all')
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/description.png)

#### *Notice:* 
1. abnormal value for age :95
2. distribution of coupon_used_in_last6_monthis discrete，try binning this variable
```
sum(pdd.duplicated())
```
#### No duplicates!

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
#### 11.6% of customers used coupons before - acceptable!

#### Numeric Variable
1. age
```
pdd['age'].plot(kind='hist', bins=40)
```
![image](https://github.com/cassiezy/Coupon-Usage-Prediction/blob/master/pic/1.png)

#### *Notice: majority are aged 25-60, try binning them later* 

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

#### *Notice: long tail, need to find out the outliers* 
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
#### *80% of users didn't use coupons last month.*

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
2 rows × 21 columns
































