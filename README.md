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

#### *Notice: 1. abnormal value for age :95
####          2. distribution of coupon_used_in_last6_monthis discrete，try binning this variable
```
sum(pdd.duplicated())
```
#### No duplicates!


