##A)Carregar llibreries
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

##B)Carregar el dataset
df = pd.read_csv('SpotifyFeatures.csv')
print("Mostra:", df.head()) #Mostra 5 rows
print("Info:", df.info()) #Mostra el tipus de columns
print("Shape:", df.shape) 

##C)Preparar dades
#1) Tractar duplicats
#Quantitat duplicats
print("\nQuantitat de dupplicats:", df.duplicated(subset="track_id", keep="first").sum())

#Treure duplicats
df.drop_duplicates(subset="track_id", keep="first", inplace=True) #Elimina duplicats segons track_id modificant df og
print("Quantitat de dupplicats després:", df.duplicated(subset="track_id", keep="first").sum())
print("Nova shape sense duplicats:", df.shape) #Mostra rowsxcolumns

#Treure columna "genre" (els duplicats eren perquè una canço pot tenir més d'un genre)
df.drop("genre", axis=1, inplace=True)
print("Nova shape sense genre:", df.shape) #Mostra rowsxcolumns

#2) Corretgir valors errònis
#Comprovar valors de key, mode i time_signature
print("\nKey values:", df['key'].value_counts())
print("Mode values:", df["mode"].value_counts())
print("Time_signature values:", df["time_signature"].value_counts())

#Eliminar 0/4
print("Quantitat de '0/4'abans:", (df["time_signature"] == "0/4").sum()) 
df = df[df["time_signature"]!="0/4"]
print("Quantitat de '0/4' després:", (df["time_signature"] == "0/4").sum())
print("New shape sense '0/4':", df.shape)

#ALTERNATIVA
#Substituir 0/4 (impossible) per 4/4 (més comú)
#df["time_signature"] = df['time_signature'].replace('0/4','4/4')
#print("Nous valors de time_signature:", df["time_signature"].value_counts())

#Comprovar i eliminar NaN
print("Quantitat de Nan abans:", df.isna().sum().sum()) #Quantitat Nan en tot df
#print("NaN per features:", df.isna().sum())
df.dropna(axis=0, inplace=True) #Files on hi ha algún NaN
print("Quantitat de Nan després:", df.isna().sum().sum())
print("New shape sense Nan:", df.shape)

#3) Eliminar columnes innecesàries
#Drop columnes: 'artist_name','track_name','track_id 
df.drop(['artist_name','track_name','track_id'], axis=1, inplace=True) #Elimina les columnes modificant df og
print("\nNova shape sense artist_name, track_name, track_id:", df.shape) 

#4)Codificar columnes no numèriques
#METODE1
df1 = df.copy()
#Diccionaris codificació valors únics
mode_dict = {'Major' : 1, 'Minor' : 0}
key_dict = {'C' : 1, 'C#' : 2, 'D' : 3, 'D#' : 4, 'E' : 5, 'F' : 6, 
        'F#' : 7, 'G' : 9, 'G#' : 10, 'A' : 11, 'A#' : 12, 'B' : 12}
time_signature_dict = {'1/4' : 1, '3/4' : 3, '4/4' :4, "5/4": 5}
#Codificar columna key 
df1["key"] = df1["key"].map(key_dict)
#Codificar columna mode 
df1["mode"] = df1["mode"].map(mode_dict)
#Codificar columna time_signature
df1["time_signature"] = df1["time_signature"].map(time_signature_dict)

#METODE2
df2 = df.copy()
#Codificar columna key
freq = df2["key"].value_counts(normalize=True)
df2["key"] = df2["key"].map(freq)
#Codificar columna mode 
freq = df2["mode"].value_counts(normalize=True)
df2["mode"] = df2["mode"].map(freq)
#Codificar columna time_signature
freq = df2["time_signature"].value_counts(normalize=True)
df2["time_signature"] = df2["time_signature"].map(freq)

##D)Establir model
#1) Bucle per tipus de codificació
datasets = [df1, df2]
#data = df.copy()
for data in datasets: 

    #2) Calcular k millor
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,30), timings=False, metric='distortion')
    visualizer.fit(data)       
    #visualizer.show()        
    k = visualizer.elbow_value_ #millor k

    #3) Fer amb Scaler (mateix rang totes feature)
    scaler = StandardScaler()
    data_scal = scaler.fit_transform(data)

    #4) Entrenar i predir 
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    pred = kmeans.fit_predict(data_scal)

    ##Visualitzar pred
    #1) Reduir components a 2 amb PCA
    pca = PCA(n_components=2)
    data_scal_2 = pca.fit_transform(data_scal)

    #Mirar la rellevàcia de cada feature en el càlcul de la component i la variancia per component
    #pd.DataFrame(data=pca.components_, columns=data.columns, index=['C1', 'C2'])
    #print(pca.explained_variance_ratio.cumsum())

    #Crear dataframe per visualitzar 
    pca_2 = pd.DataFrame(data_scal_2, columns=['Component 1', 'Component 2'])
    pca_2["cluster"] = pred #Mateix que: kmeans.labels_

    #2) Scatterplot (en jupyter)
    """Plot
    center = pca.transform(kmeans.cluster_centers_)
    sns.scatterplot(data=pca_2, x='Component 1', y='Component 2', hue='cluster', palette='viridis')##Plot les dades test
    plt.scatter([c[0] for c in center],[c[1] for c in center], marker='X', linewidths=3, color='red')
    plt.show()
    """
    #Evaluar: higher better - lower better
    print("\n", calinski_harabasz_score(data_scal, kmeans.labels_), davies_bouldin_score(data_scal, kmeans.labels_))
