# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score

# load data

# voce pode definir funcoes para facilitar a utilizacao da mesma analise futuramente ou por 
# outras pessoas
# mesmo que vc utilize essa funcao apenas uma vez neste script, ter ela definida eh uma boa pratica 
# em DS. Nos desenvolvemos em notebook mas depois pra colocar o modelo em producao a gente 
# implementa por meio de scripts python aproveitando as funcoes que desenvolvemos no notebook

def load_data(path):
    
    df = pd.read_csv(path, sep = ';')
    
    return df

path = 'https://github.com/knmukai/retail-customer-segmentation-analysis/blob/main/marketing_campaign.csv?raw=true'
df = load_data(path)

def visualize_df_data(df):    
    print("preview dos dados")
    print(df.head(5))
    print("tipo dos dados")
    print(df.info())

visualize_df_data(df)

# data preprocessing


def get_processed_df(df, columns_to_drop):
    
    df_processed = df.drop(columns_to_drop, axis=1) #drop colunas desnecessarias
    
    return df_processed

columns_to_drop = ['ID', 'Dt_Customer', 'Recency', 'Complain', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Z_CostContact', 'NumDealsPurchases','NumWebPurchases', 'Z_Revenue',
'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

df_processed = get_processed_df(df, columns_to_drop)

# df_processed = df

# df_processed = df_processed.drop(['ID', 'Dt_Customer', 'Recency', 'Complain', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
# 'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Z_CostContact', 'NumDealsPurchases','NumWebPurchases', 'Z_Revenue',
# 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'], axis=1) #drop colunas desnecessarias

def get_df_stats(df):
    
    print("colunas utilizadas")
    print(df.info())

    print("caracteristica das variaveis")
    df.select_dtypes('number').describe().transpose() #caracteristica das variaveis numericas

    print("visulizar colunas com valores faltantes")
    print(df.isna().sum()) #Income com valores faltantes
    
get_df_stats(df_processed)

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

#############################################################################################
# Avaliar se esse bloco de codigo poderia estar dentro da funcao get_processed_df
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
#############################################################################################

sns.set(font_scale=0.6) #diminuir tamanho da fonte
plt.subplot(2, 2, 1)
sns.histplot(x = df_processed["Year_Birth"])
plt.subplot(2, 2, 2)
sns.histplot(x = df_processed["Income"])
plt.show()

print("codificacao das colunas categoricas")
df_encoded = df_processed.copy()
le = LabelEncoder() 
df_encoded['Education'] = le.fit_transform(df_encoded['Education'])
df_encoded['Marital_Status'] = le.fit_transform(df_encoded['Marital_Status'])
print(df_encoded.head(5))

print("normatização")
min_max_scaler = MinMaxScaler()
df_encoded[df_encoded.columns] = min_max_scaler.fit_transform(df_encoded)
print(df_encoded.head(5))

print("definição do numero de clusters")
print("elbow method")


kmeans = KMeans(random_state=42)
elb_visualizer = KElbowVisualizer(kmeans, k=(1,15))
elb_visualizer.fit(df_encoded)    
elb_visualizer.show() 

print("silhouette method")
silhouette_coefficients = []

for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df_encoded)
    score = silhouette_score(df_encoded, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.plot(range(2, 15), silhouette_coefficients)
plt.xticks(range(2, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

kmeans = KMeans(n_clusters = 3, random_state=42)
sil_visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
sil_visualizer.fit(df_encoded)    
sil_visualizer.show() 
print("pelo elbow k=4, pelo silhouette k=2. Definido k=3")

print("k-means clustering")
kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(df_encoded)
df_processed['cluster']=kmeans.labels_
print(df_processed.head(5))

x=df_encoded.iloc[:,2]
y=df_encoded.iloc[:,3]
z=df_encoded.iloc[:,4]
fig = plt.figure(figsize=(20,10))
plot3d = fig.add_subplot(111, projection='3d')
plot3d.scatter(x,y,z, c= kmeans.labels_)
plt.show()
