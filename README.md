# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
## FEATURE SCALING:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
df.dropna()
max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/8881d48d-05dc-4657-8943-28b5384a2f99)

![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/e588d9c6-7f1e-4bb5-9258-2f91f9be053d)

![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/cfa8bc2a-fc8f-4758-9bd0-5fae4723bbd4)

### STANDARD SCALING
```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/f0adbf5c-2446-4a99-bca5-e3590d3744bb)
### MIN-MAX SCALING:
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/132ba4cb-aa1b-4036-b435-5d2c34d59890)

### MAXIMUM ABSOLUTE SCALING:
```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/95602b90-ba66-4103-9a38-9b121fc6ed32)
### ROBUST SCALING:
```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/b5087b76-8145-448c-9d64-b16de057321c)
   ### The best feature scaling methods for 'bmi' dataset could be MinMax Scaling.MinMax Scaling scales the data to a fixed range, preserving the original range of the data.While MinMax Scaling is sensitive to outliers, it can still be effective if your dataset does not contain extreme outliers.
## FEATURE SELECTION:
```
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv",na_values=[" ?"])
df=df.dropna(axis=0)
df
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/23a6d5f6-96c2-4089-8eea-69c963e815d2)

### FILTER METHOD:
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 5
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
k_anova = 5  
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
selected_features_anova = X.columns[selector_anova.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/216ec88f-dafc-47a9-b084-a3d6fa4e7b3a)

### WRAPPER METHOD:
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 5
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/bb3b3c6a-d6da-4d56-ac65-c00efefc059a)

### EMBEDDED METHOD:
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModeL
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
selector = SelectFromModel(estimator=logreg_l1, threshold=None)
selector.fit(X, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_features_indices]
print("Selected features using embedded method (Lasso regularization):")
print(selected_features)
```
![image](https://github.com/Kirthi-Niharika/EXNO-4-DS/assets/114135005/13239ff5-1b9e-4f02-94fa-15a1a1e44295)

###   For this 'income' dataset, Embedded Method is best for feature selection.Embedded methods provide feature importance scores directly within the context of the chosen model. Embedded methods automatically select relevant features during model training, which can be beneficial for reducing overfitting and improving generalization performance

# RESULT:
     Thus, Feature selection and Feature scaling has been performed using the given dataset.
