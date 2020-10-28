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
no missing value
