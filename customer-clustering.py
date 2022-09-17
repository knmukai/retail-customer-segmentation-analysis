# import libraries
import pandas as pd
import seaborn as sns
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> ed711e9abce2110b1b506118218d873532ebe9ed
import numpy as np

# load data
path = 'https://github.com/knmukai/retail-customer-segmentation-analysis/blob/main/marketing_campaign.csv?raw=true'
df = pd.read_csv(path, sep = ';')
<<<<<<< HEAD
print("preview dos dados")
print(df.head(5))
print("tipo dos dados")
=======
print(df.head(5))
>>>>>>> ed711e9abce2110b1b506118218d873532ebe9ed
print(df.info())

# data preprocessing
df_processed = df

<<<<<<< HEAD
df_processed = df_processed.drop(['Dt_Customer', 'Recency', 'Complain', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Z_CostContact', 'Z_Revenue', ], axis=1) #drop colunas desnecessarias
print("colunas utilizadas")
print(df_processed.info())

print("caracteristica das variaveis")
df_processed.select_dtypes('number').describe().transpose() #caracteristica das variaveis numericas

print("visulizar colunas com valores faltantes")
print(df_processed.isna().sum()) #Income com valores faltantes

print("visualizar caracteristica dos clientes")
sns.set(font_scale=0.6) #diminuir tamanho da fonte
plt.subplot(3, 2, 1)
sns.histplot(x = df_processed["Year_Birth"])
plt.subplot(3, 2, 2)
sns.histplot(x = df_processed["Education"])
plt.subplot(3, 2, 3)
sns.histplot(x = df_processed["Marital_Status"])
plt.subplot(3, 2, 4)
sns.histplot(x = df_processed["Income"])
plt.subplot(3, 2, 5)
sns.histplot(x = df_processed["Kidhome"])
plt.subplot(3, 2, 6)
sns.histplot(x = df_processed["Teenhome"])
plt.show()

print("media da renda por nivel de educacao")
print(df.groupby(['Education'])['Income'].mean())
df['Income'].fillna(df.groupby('Education')['Income'].transform('mean'), inplace = True) #substituir valores na pela media por Education
print("valores faltantes de renda substituidos pela media por nivel de educacao")
print(df_processed.isna().sum()) #Income com valores faltantes

# remoção de outliers


#df_dummies = pd.get_dummies(data=df_processed, drop_first=True)
#print(df_dummies.info())
#print(df_dummies.head())
=======
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
>>>>>>> ed711e9abce2110b1b506118218d873532ebe9ed