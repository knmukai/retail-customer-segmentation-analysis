# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# load data
path = 'https://github.com/knmukai/retail-customer-segmentation-analysis/blob/main/marketing_campaign.csv?raw=true'
df = pd.read_csv(path, sep = ';')
print("preview dos dados")
print(df.head(5))
print("tipo dos dados")
print(df.info())

# data preprocessing
df_processed = df

df_processed = df_processed.drop(['ID', 'Dt_Customer', 'Recency', 'Complain', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Z_CostContact', 'NumDealsPurchases','NumWebPurchases', 'Z_Revenue',
'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'], axis=1) #drop colunas desnecessarias
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
#plt.show()

print("substituir status marital Alone, Absurd e YOLO por Single e Widow por Divorced")
print(df_processed['Marital_Status'].value_counts())
df_processed['Marital_Status'].replace(['Alone','Absurd','YOLO'], 'Single', inplace=True)
df_processed['Marital_Status'].replace(['Widow'], 'Divorced', inplace=True)
print(df_processed['Marital_Status'].value_counts())

print('sinalizar apenas se possui ou não filhos')
df_processed['Childhome'] = np.where((df_processed['Kidhome'] == 0) | (df_processed['Teenhome'] == 0), 0, 1)
df_processed = df_processed.drop(['Kidhome', 'Teenhome'], axis=1) #drop colunas de filhos
print(df_processed['Childhome'].value_counts())

print("media da renda por nivel de educacao")
print(df.groupby(['Education'])['Income'].mean())
df['Income'].fillna(df.groupby('Education')['Income'].transform('mean'), inplace = True) #substituir valores na pela media por Education
print("valores faltantes de renda substituidos pela media por nivel de educacao")
print(df_processed.isna().sum()) #Income com valores faltantes

# remoção de outliers
print("qtd de registros antes da remocao de outliers:"+str(len(df_processed)))
df_processed = df_processed[(df_processed['Year_Birth']>=1940) & (df_processed['Income']<=100000)].reset_index()
print("qtd de registros antes da remocao de outliers:"+str(len(df_processed)))
sns.set(font_scale=0.6) #diminuir tamanho da fonte
plt.subplot(2, 2, 1)
sns.histplot(x = df_processed["Year_Birth"])
plt.subplot(2, 2, 2)
sns.histplot(x = df_processed["Income"])
#plt.show()

print("codificacao das colunas categoricas")
df_encoded = df_processed
le = LabelEncoder() 
df_encoded['Education'] = le.fit_transform(df_encoded['Education'])
df_encoded['Marital_Status'] = le.fit_transform(df_encoded['Marital_Status'])
print(df_encoded.head(5))

print("normatização")

#df_dummies = pd.get_dummies(data=df_processed, drop_first=True)
#print(df_dummies.info())
#print(df_dummies.head())
