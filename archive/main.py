##Carregar llibreries
import pandas as pd         ###py -m pip install pandas

from sklearn.decomposition import PCA ###py -m pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer ##py -m pip install yellowbrick

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
#1) Dividir train i test
#Dividir train i test
data = df.copy()
x_train, x_test = train_test_split(data, test_size = 0.2, random_state = 0)

#2) Calcular k millor
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,30), timings=False, metric='distortion')
visualizer.fit(x_train)       
#visualizer.show()        
k = visualizer.elbow_value_ #millor k

#3) Fer amb Scaler (mateix rang totes feature) i KMeans 
scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train)
x_test_scal = scaler.transform(x_test)

kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)

#4) Entrenar amb train i predir test
pred_train =kmeans.fit_predict(x_train_scal)
pred_test = kmeans.predict(x_test_scal)

##Visualitzar pred
#1) Reduir components a 2 amb PCA
pca = PCA(n_components=2)
x_train_2 = pca.fit_transform(x_train_scal)
x_test_2 = pca.transform(x_test_scal)

#Mirar la rellevàcia de cada feature en el càlcul de la component i la variancia per component
#pd.DataFrame(data=pca.components_, columns=x_train_scal.columns, index=['C1', 'C2'])
#pca.explained_variance_ratio.cumsum()

#Crear dataframe per visualitzar per train i test
df_train = pd.DataFrame(x_train_2, columns=['Component 1', 'Component 2'])
df_train["cluster"] = pred_train

df_test = pd.DataFrame(x_test_2, columns=['Component 1', 'Component 2'])
df_test["cluster"] = pred_test

#2) Scatterplot (en jupyter)
##Plot les dades train
"""
center = pca.transform(kmeans.cluster_centers_)
sns.scatterplot(data=df_train, x='Component 1', y='Component 2', hue='cluster', palette='viridis')##Plot les dades test
plt.scatter([c[0] for c in center],[c[1] for c in center], marker='X', linewidths=3, color='red')
plt.show()

##Plot les dades test
sns.scatterplot(data=df_test, x='Component 1', y='Component 2', hue='cluster', palette='viridis')##Plot les dades test
plt.scatter([c[0] for c in center],[c[1] for c in center], marker='X', linewidths=3, color='red')
plt.show()

"""

