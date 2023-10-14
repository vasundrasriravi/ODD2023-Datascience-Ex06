# ODD2023-Datascience-Ex06
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## Algorithm:
- Step1: Read the given Data.
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature Transformation techniques to all the features of the data set.
- Step4: Print the transformed features.
## Program:
```
Developed By: VASUNDRA SRI R
Register No: 212222230168
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
## OUTPUT:
### Original Data:
![o1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/bab57511-8aba-44b6-8298-8d0ca52944b1)

### Data information:
![info](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/2326afee-2a77-4ad2-a8d2-39db98fe6aee)

### Data describe:
![o2](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/f0f00a90-4e2c-41eb-9e6c-7f785fa216eb)

### Before transformation:
![t1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/6f56f6ff-0b9e-4c8b-8c4e-0d429c7a1a5e)
![t2](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/279d95ce-01b5-4709-8f18-2d36786a91a8)
![t3](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/59001803-0b18-4d78-86d8-131154af9669)
![t4](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/93c46035-3627-4fa3-a5cd-d230f9c60f0e)

### Log transformation:
![l1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/98cae94c-b78e-45c9-9381-23139b2b21e6)

### Reciprocal transformation:
![r1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/f52abf4f-4e78-48de-b81a-7f30b21a6c13)

### Square root transformation:
![s1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/9efd3129-1bfd-49e1-a9f4-88f50edcc78e)
![s2](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/db524b2a-4a73-473f-97a4-1f7fa9f3264f)

### Power transformation:
![p1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/2d18c479-4e44-473f-a39c-c819e4bd7fdb)
![p2](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/1602396c-abeb-48e5-82ad-3e19e5d34e85)

### Quantile transformation:
![q1](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex06/assets/119393983/771d3daa-fa32-407f-83d4-6b942b6a988d)

# RESULT:
Thus feature transformation is done for the given dataset.
