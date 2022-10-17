# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from sklearn.metrics import r2_score, mean_squared_error

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
print("qtd de registros depois da remocao de outliers:"+str(len(df_processed)))

cols=['MntWines', 'MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']

#for k in range(0, len(cols)):
#    plt.subplot(2, 3, k + 1)
#    sns.boxplot(y = df_processed[cols[k]])
#plt.show()

df_processed.insert(12,'MntTotal',df_processed['MntWines']+df_processed['MntFruits']+df_processed['MntMeatProducts']+df_processed['MntFishProducts']+df_processed['MntSweetProducts']
+df_processed['MntGoldProds'])
df_processed = df_processed.drop(['MntWines', 'MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds'], axis=1) 

df_processed.insert(6,'age', date.today().year-df_processed['Year_Birth'])
df_processed = df_processed.drop(['Year_Birth'], axis=1) 
df_processed = df_processed.drop(['index'], axis=1) 

print(df_processed.head(5))

df_dummies = pd.get_dummies(data=df_processed, drop_first=True)
print(df_dummies.head(5))

plt.figure()
sns.heatmap(df_dummies.corr(), annot=True)
plt.show()

features = df_dummies.columns.drop('MntTotal')
target = ['MntTotal']

X = df_dummies[features].values
y = df_dummies[target].values
split_test_size = 0.10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state=8)

print('Variáveis explicativas de treino:', X_train.shape)
print('Variáveis explicativas de teste:', X_test.shape)
print('Variável alvo de treino:', y_train.shape)
print('Variável alvo de teste:', y_test.shape)

print("normatização")

fscaler = MinMaxScaler()
X_train = fscaler.fit_transform(X_train)
X_test = fscaler.transform(X_test)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('mean square error: ', mean_squared_error(y_test, predictions, squared=False)) #0+ (lower is better)
print('mean r2 score: ', r2_score(y_test, predictions)) #0-1 (larger is better)
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=predictions)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_)
print(pd.DataFrame(model.coef_.round(2).transpose(),features,columns=['Coefficient']))
predictions = model.predict(X_test)

print('mean square error: ', mean_squared_error(y_test, predictions, squared=False)) #0+ (lower is better)
print('mean r2 score: ', r2_score(y_test, predictions)) #0-1 (larger is better)

plt.subplot(1, 2, 2)
sns.regplot(x=y_test, y=predictions)
plt.show()

