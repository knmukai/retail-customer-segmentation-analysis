# import libraries
import pandas as pd
import seaborn as sns
import numpy as np

# load data
path = 'https://github.com/knmukai/retail-customer-segmentation-analysis/blob/main/marketing_campaign.csv?raw=true'
df = pd.read_csv(path, sep = ';')
print(df.head(5))
print(df.info())

# data preprocessing
df_processed = df

print(df_processed.isna().sum()) #Income com valores faltantes

print(df_processed.groupby(['Education'])['Education'].count())
print(df_processed.groupby(['Marital_Status'])['Marital_Status'].count())

print(df.groupby(['Education'])['Income'].mean())
df['Income'].fillna(df.groupby('Education')['Income'].transform('mean'), inplace = True) #substituir valores na pela media por Education

df_processed = df_processed.drop(['Dt_Customer', 'Recency', 'Complain', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Z_CostContact', 'Z_Revenue', ], axis=1) #drop colunas desnecessarias
print(df_processed.info())

df_dummies = pd.get_dummies(data=df_processed, drop_first=True)
print(df_dummies.info())
print(df_dummies.head())