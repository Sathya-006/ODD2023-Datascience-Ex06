# ODD2023-Datascience-Ex06
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1: Read the given Data

STEP 2: Clean the Data Set using Data Cleaning Process

STEP 3: Apply Feature Transformation techniques to all the features of the data set

STEP 4: Print the transformed features

# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/d60ea735-f1d2-444c-b727-7c53103e6fa6)
```
df.head()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/ce7dbbf5-d420-499d-960f-413c3f5119e5)
```
df.isnull().sum()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/c0e0bf3d-d6d9-4ddd-a64f-49a23fb9271d)
```
df.info()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/85546518-22cf-4f30-aa9b-015348664a6a)
```
df.describe()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/33c3221f-d83b-4899-80cd-323d4ce73a80)
```
df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/d05e4ed0-d7a3-45a3-88f2-9b1925b066d5)
```
sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/d3a32a49-dc4b-4106-87ab-6592e39025bb)
```

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/7e892b23-99b4-469c-b2e5-dcc724274eb5)
```
sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/079582d5-86bf-4367-896a-a0237a45b744)
```
df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/f0488283-9c24-4793-8196-bd3ce99e03c3)
```
df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/f417620a-a538-404a-9f3f-06296b83d9fa)
```
df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/47c9e84e-cf93-4f58-a668-ad811bf82a7b)
```
df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/bc427812-c54d-4d61-b8aa-c1c2092f7e9a)
```
from sklearn.preprocessing import PowerTransformer
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/5b5af673-a9af-4976-8709-8726f960278f)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex06/assets/121661327/d5dee3f6-adbc-48d4-a0af-242bb8ec3404)

# Result
Thus feature transformation is done for the given set
