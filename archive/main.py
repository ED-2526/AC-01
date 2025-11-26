##Carregar llibreries
import pandas as pd ###py -m pip install pandas
import numpy as np ###py -m pip install numypy

from sklearn.decomposition import PCA ###py -m pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from yellowbrick.cluster import KElbowVisualizer ##py -m pip install yellowbrick
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

import seaborn as sns ##py -m pip install seaborn

import matplotlib.pyplot as plt ##py -m pip install matplotlib

##Carregar el dataset
df = pd.read_csv('SpotifyFeatures.csv')
print(f"Mostra: {df.head()}")
print(f"Info: {df.info()}")

##Preparar dades
#1) Treure cançons duplicades
#Quantitat duplicats
print("Quantitat de dupplicats:", df.duplicated(subset="track_id", keep="first").sum())

#Treure duplicats
df = df.drop_duplicates(subset="track_id", keep="first")
print(f"Nova shape: {df.shape}")

#2) Eliminar columnes innecesàries
#Columnes: 'artist_name','track_name','track_id 
df.drop(['artist_name','track_name','track_id'], axis=1, inplace=True) 
print(f"Nova shape: {df.shape}")

#3) Corretgir valors errònis (impossibles)
#Comprovar valors de key, mode i time_signature
print("Key values:", df['key'].value_counts())
print("Mode values:", df["mode"].value_counts())
print("Time_signature values:", df["time_signature"].value_counts())

#Substituir 0/4 (impossible) per 4/4 (més comú)
df["time_signature"] = df['time_signature'].replace('0/4','4/4')
print("Nova shape:", df["time_signature"].value_counts())

#4) Codificar columnes no numèriques
#Codificar columna key 
freq = df["key"].value_counts(normalize=True)
df["key"] = df["key"].map(freq)

#Codificar columna mode 
freq = df["mode"].value_counts(normalize=True)
df["mode"] = df["mode"].map(freq)

#Codificar columna time_signature
freq = df["time_signature"].value_counts(normalize=True)
df["time_signature"] = df["time_signature"].map(freq)

#Codificar columna time_signature
freq = df["genre"].value_counts(normalize=True)
df["genre"] = df["genre"].map(freq)
print("Mostra:", df.head())

##Establir model
#1) Copiar df
data = df.copy()

#2) Calcular k millor
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,30), timings=False, metric='distortion')
visualizer.fit(data)       
#visualizer.show()        
k = visualizer.elbow_value_ #millor k

#3) Fer amb Scaler (mateix rang totes feature) i KMeans 
scaler = StandardScaler()
data_scal = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)

#4) Entrenar i predir 
pred = kmeans.fit_predict(data_scal)

##Visualitzar pred
#1) Reduir components a 2 amb PCA
pca = PCA(n_components=2)
data_2 = pca.fit_transform(data_scal)

#Mirar la rellevàcia de cada feature en el càlcul de la component i la variancia per component
#pd.DataFrame(data=pca.components_, columns=data.columns, index=['C1', 'C2'])
#pca.explained_variance_ratio.cumsum()

#Crear dataframe per visualitzar 
df_2 = pd.DataFrame(data_2, columns=['Component 1', 'Component 2'])
df_2["cluster"] = pred

#2) Scatterplot (en jupyter)
##Plot les dades train
"""
center = pca.transform(model.cluster_centers_)
sns.scatterplot(data=df_2, x='Component 1', y='Component 2', hue='cluster', palette='viridis')##Plot les dades test
plt.scatter([c[0] for c in center],[c[1] for c in center], marker='X', linewidths=3, color='red')
plt.show()
"""
#Evaluat: higher better - lower better
print(calinski_harabasz_score(data, kmeans.labels_), davies_bouldin_score(data, kmeans.labels_))
