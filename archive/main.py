"""
Docstring for main
genre: The genre in which the track belongs
artist_name: The artists' names who performed the track
track_name: Name of the track
track_id: The Spotify ID for the track
popularity: The popularity of a track is a value between 0 and 100
acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic
danceability: Danceability describes how suitable a track is for dancing based
duration_ms: The track length in milliseconds
energy: Measure of intensity and activity from 0.0 to 1.0
instrumentalness: Predicts whether a track contains no vocals (nearer 1=no vocals)
key: The key the track is in
liveness: Detects the presence of an audience in the recording (higher 0.8=live)
loudness: The overall loudness of a track in decibels (dB)
mode: Mode indicates the scale (major or minor) of a track
speechiness: Speechiness detects the presence of spoken words in a track
tempo: The overall estimated tempo of a track in beats per minute
time_signature: An estimated time signature
valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track
"""

##A)Carregar llibreries
import pandas as pd ###py -m pip install pandas
import numpy as np ###py -m pip install numypy

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD ###py -m pip install scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, MiniBatchKMeans 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer ##py -m pip install yellowbrick
import seaborn as sns ##py -m pip install seaborn
import random
import matplotlib.pyplot as plt ##py -m pip install matplotlib
from scipy.spatial.distance import euclidean

##B)Carregar el dataset
df = pd.read_csv('C:\\Users\\david\\Desktop\\AC\\AC-01-1\\archive\\SpotifyFeatures.csv')
#print(df.head()) #Mostra 5 rows
print(df.info()) #Mostra el tipus de columns: df.dtypes

#2) Corretgir valors errònis (impossibles)
#Comprovar valors de key, mode i time_signature
#print("Key values:", df['key'].value_counts())
#print("Mode values:", df["mode"].value_counts())
#print("Time_signature values:", df["time_signature"].value_counts())
#print("Genre values::", df["genre"].value_counts())

#Eliminar key: "0/4" (impossible)
print("Quantitat de '0/4'abans:", (df["time_signature"] == "0/4").sum()) #Quantitat key: "0/4"
df = df[df["time_signature"] != "0/4"]
print("Quantitat de '0/4' després:", (df["time_signature"] == "0/4").sum())
print("New shape sense '0/4':", df.shape)

#Substituir Children’s Music per Children's Music 
print("Quantitat de Children’s Music abans:", (df["genre"] == "Children’s Music").sum()) 
print("Quantitat de Children's Music abans:", (df["genre"] == "Children's Music").sum()) 
df.loc[df["genre"]=="Children’s Music", "genre"] = "Children's Music"
print("Quantitat de Children’s Music desrpés:", (df["genre"] == "Children’s Music").sum()) 
print("Quantitat de Children's Music després:", (df["genre"] == "Children's Music").sum()) 
print("New valors de genre sense Children’s Music :", df["genre"].nunique())

#Eliminar NaN
print("Quantitat de Nan abans:", df.isna().sum().sum()) #Quantitat Nan en tot df
#print("NaN per features:", df.isna().sum())
df.dropna(axis=0, inplace=True) #
print("Quantitat de Nan després:", df.isna().sum().sum())
print("New shape sense Nan:", df.shape)

#3) Eliminar columnes 
#Reiniciar index en dataframe
df.reset_index(drop=True, inplace=True)

#Guardar columnes: 'track_name', 'artist_name'
canciones = df[["track_name", "artist_name"]]

#track_name = df[["track_name"]]
#artist_name = df[["artist_name"]]

#Drop olumnes: 'artist_name','track_name','track_id 
df.drop(['artist_name','track_name','track_id'], axis=1, inplace=True) #Elimina les columnes modificant df og
print(f"Nova shape sense artist_name, track_name, track_id: {df.shape}") #Mostra rowsxcolumns

num_col = df.select_dtypes(exclude="object").columns
cat_col = df.select_dtypes(include="object").columns

#4)Codificar columnes no numèriques
#Genre amb OneHotEncoder i resta amb frequencia
df1 = df.copy(deep=True)
freq = df1["key"].value_counts(normalize=True)
df1["key"] = df1["key"].map(freq)
#Codificar columna mode 
freq = df["mode"].value_counts(normalize=True)
df1["mode"] = df1["mode"].map(freq)
#Codificar columna time_signature
freq = df1["time_signature"].value_counts(normalize=True)
df1["time_signature"] = df1["time_signature"].map(freq)
#Codificar columna genre
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_temp = df1[['genre']].copy()
encoded_array = enc.fit_transform(df_temp)
feature_names = enc.get_feature_names_out(['genre'])
encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df1.index)
df1 = pd.concat([df1.drop(['genre'], axis=1), encoded_df], axis=1)
print("Nova shape df1:", df1.shape)
print("Noves columnas:", df1.columns)

#5) Mirar histograma
df1.hist(figsize=(30, 30)), plt.show()
#6) Mirar correlació (pearson)
corr_matrix3 = df1[num_col].corr()
plt.figure(figsize=(12, 12))
ax = sns.heatmap(corr_matrix3, annot=True, cmap='coolwarm')
plt.show()

#7) Escalar columnes
scaler = StandardScaler()
data1_scal = scaler.fit_transform(df1)
#df1_scal = pd.DataFrame(data=scaler.fit_transform(data), columns=data.columns)

#DF1 KMEANS
#1) Calcular k millor
model = KMeans(n_init=10, random_state=0)
visualizer = KElbowVisualizer(model, k=(2,30), timings=False, metric='distortion') #default distance_metric: euclidean
visualizer.fit(data1_scal)       
#visualizer.show() #Quitar en py
#visualizer.show(outpath="kmeans.png") #Guardar gráfica
k = visualizer.elbow_value_ #millor k
#2) Entrenar i predir amb KMEANS
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
#kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0)
kmeans1 = kmeans.fit_predict(data1_scal) #Mateix que: kmeans.labels__
#data_scal["cluster"] = pred

#DF1 DBSCAN
#1) Calcular eps
min_samples = 2 * data1_scal.shape[1]
nn = NearestNeighbors(n_neighbors=min_samples).fit(data1_scal) #default algorithm: auto; most appropiate, default metric: minkowski
dists, _ = nn.kneighbors(data1_scal)
kdist = np.sort(dists[:, -1])
plt.figure(figsize=(10,4))
plt.plot(kdist, label=f'{min_samples}-distance sorted')
plt.title('k-distance plot')
plt.xlabel('Observaciones (ordenadas)')
plt.ylabel(f'{min_samples}-distance')
plt.savefig("dbscan.png")
plt.show()
print(kdist[175000])
#2) Entrenar i predir amb DBSCAN
min_samples = 2 * data1_scal.shape[1]
dbscan = DBSCAN(eps=3.8, min_samples=min_samples, n_jobs=-1,) #default algorithm: auto of NearestNeighbours; most appropiate
dbscan1 = dbscan.fit_predict(data1_scal)
#print(calinski_harabasz_score(data1_scal, dbscan1), davies_bouldin_score(data1_scal, dbscan1))

#DF1 HDBSCAN
#1) Entrenar i predir amb HDBSCAN
min_samples = 2 * data1_scal.shape[1]
hdbscan = HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples, n_jobs=-1, store_centers="centroid") #default metric: euclidean
hdbscan1 = hdbscan.fit_predict(data1_scal)

#EVALUACIO MODELS 
#KMEANS
print(calinski_harabasz_score(data1_scal, kmeans1), davies_bouldin_score(data1_scal, kmeans1))
#DBSCAN
print(calinski_harabasz_score(data1_scal, dbscan1), davies_bouldin_score(data1_scal, dbscan1))
#HDBSCAN
print(calinski_harabasz_score(data1_scal, hdbscan1), davies_bouldin_score(data1_scal, hdbscan1))

len(set(kmeans1)), len(set(dbscan1)), len(set(hdbscan1))

# Crear sampled dataset per proves més ràpides
# Samplearem 15% de les dades per a proves
sample_size = int(len(df) * 0.15)
#sample_size = int(len(df) * 0.15 / len(set(hdbscan1)))

#df1_hdbscan1 = pd.DataFrame(data=data1_scal, columns=df1.columns)
#df1_hdbscan1["label"] = hdbscan1
#sample_indices = df1_hdbscan1.groupby("label").apply(lambda x: x.sample(sample_size, random_state=0) if len(x) >= sample_size else x, include_groups=False).droplevel("label").index
sample_indices = np.random.RandomState(42).choice(len(df), sample_size, replace=False)

df1_sample = data1_scal[sample_indices]
#pred= pred[sample_indices]
print("Mida orignal:", df.shape)
print("Mida 20%:", df1_sample.shape)

#VISUALITZACIÓ RESULTATS AMB DIFERENTS MÈTODES DE REDUCCIÓ DE DIMENSIONALITAT
# 1. PCA Estàndard
pca_sample = PCA(n_components=2, random_state=42)
df1_sample_pca = pca_sample.fit_transform(df1_sample)

df1_sample_pca_2 = pd.DataFrame(data=df1_sample_pca, columns=['Component 1', 'Component 2'])
df1_sample_pca_2["kmeans"] = kmeans1[sample_indices]
df1_sample_pca_2["dbscan"] = dbscan1[sample_indices]
df1_sample_pca_2["hdbscan"] = hdbscan1[sample_indices]

# 2. KernelPCA RBF
kpca_sample = KernelPCA(n_components=2, kernel='rbf', random_state=42, n_jobs=-1)
df1_sample_kpca = kpca_sample.fit_transform(df1_sample)

df1_sample_kpca_2 = pd.DataFrame(data=df1_sample_kpca, columns=['Component 1', 'Component 2'])
df1_sample_kpca_2["kmeans"] = kmeans1[sample_indices]
df1_sample_kpca_2["dbscan"] = dbscan1[sample_indices]
df1_sample_kpca_2["hdbscan"] = hdbscan1[sample_indices]

# 3. TruncatedSVD
tsvd_sample = TruncatedSVD(n_components=2, random_state=42)
df1_sample_tsvd = tsvd_sample.fit_transform(df1_sample)

df1_sample_tsvd_2 = pd.DataFrame(data=df1_sample_tsvd, columns=['Component 1', 'Component 2'])
df1_sample_tsvd_2["kmeans"] = kmeans1[sample_indices]
df1_sample_tsvd_2["dbscan"] = dbscan1[sample_indices]
df1_sample_tsvd_2["hdbscan"] = hdbscan1[sample_indices]

# 4. TSNE
tsne = TSNE(n_components=2, perplexity=150, learning_rate='auto', random_state=42)
df1_sample_tsne = tsne.fit_transform(df1_sample)
df1_sample_tsne_2 = pd.DataFrame(data=df1_sample_tsne, columns=['Component 1', 'Component 2'])
df1_sample_tsne_2["kmeans"] = kmeans1[sample_indices]
df1_sample_tsne_2["dbscan"] = dbscan1[sample_indices]
df1_sample_tsne_2["hdbscan"] = hdbscan1[sample_indices]

#KMEANS
fig, axes = plt.subplots(2, 2, figsize=(26, 24))
# 1. PCA Estàndard
#df1_sample_pca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_pca_2, x='Component 1', y='Component 2', hue='kmeans', style='kmeans',
               palette='tab20', ax=axes[0, 0])
axes[0, 0].set_title('PCA Estàndard (df1)')
axes[0, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))
# 2. KernelPCA RBF
#df1_sample_kpca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_kpca_2, x='Component 1', y='Component 2', hue='kmeans',  style='kmeans',
               palette='tab20', ax=axes[0, 1])
axes[0, 1].set_title('KernelPCA RBF (df1)')
axes[0, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))
# 3. TruncatedSVD
#df1_sample_tsvd_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_tsvd_2, x='Component 1', y='Component 2', hue='kmeans',  style='kmeans',
               palette='tab20', ax=axes[1, 0])
axes[1, 0].set_title('TruncatedSVD (df1)')
axes[1, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))
# 4. TSNE
#df1_sample_tsne_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_tsne_2, x='Component 1', y='Component 2', hue='kmeans', style='kmeans',
               palette='tab20', ax=axes[1, 1])
axes[1, 1].set_title('TSNE (df1)')
axes[1, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.suptitle(f"KMEANS - {tsne.perplexity}")
fig.subplots_adjust(top=0.95)
plt.legend(loc=1, bbox_to_anchor=(1.2, 1))
plt.show()

#HDBSCAN
fig, axes = plt.subplots(2, 2, figsize=(26, 24))
# 1. PCA Estàndard
#df1_sample_pca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_pca_2, x='Component 1', y='Component 2', hue='hdbscan', style='hdbscan',
               palette='tab20', ax=axes[0, 0])
axes[0, 0].set_title('PCA Estàndard (df1)')
axes[0, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))
# 2. KernelPCA RBF
#df1_sample_kpca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_kpca_2, x='Component 1', y='Component 2', hue='hdbscan',  style='hdbscan',
               palette='tab20', ax=axes[0, 1])
axes[0, 1].set_title('KernelPCA RBF (df1)')
axes[0, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))
# 3. TruncatedSVD
#df1_sample_tsvd_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_tsvd_2, x='Component 1', y='Component 2', hue='hdbscan',  style='hdbscan',
               palette='tab20', ax=axes[1, 0])
axes[1, 0].set_title('TruncatedSVD (df1)')
axes[1, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))
# 4. TSNE
#df1_sample_tsne_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_tsne_2, x='Component 1', y='Component 2', hue='hdbscan', style='hdbscan',
               palette='tab20', ax=axes[1, 1])
axes[1, 1].set_title('TSNE (df1)')
axes[1, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.suptitle(f"HDBSCAN - perplexity{tsne.perplexity}")
fig.subplots_adjust(top=0.95)
plt.legend(loc=1, bbox_to_anchor=(1.2, 1))
plt.show()

#DBSCAN
fig, axes = plt.subplots(2, 2, figsize=(26, 24))
# 1. PCA Estàndard
#df1_sample_pca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_pca_2, x='Component 1', y='Component 2', hue='dbscan', style='dbscan',
               palette='tab20', ax=axes[0, 0])
axes[0, 0].set_title('PCA Estàndard (df1)')
axes[0, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))
# 2. KernelPCA RBF
#df1_sample_kpca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_kpca_2, x='Component 1', y='Component 2', hue='dbscan',  style='dbscan',
               palette='tab20', ax=axes[0, 1])
axes[0, 1].set_title('KernelPCA RBF (df1)')
axes[0, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))
# 3. TruncatedSVD
#df1_sample_tsvd_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_tsvd_2, x='Component 1', y='Component 2', hue='dbscan',  style='dbscan',
               palette='tab20', ax=axes[1, 0])
axes[1, 0].set_title('TruncatedSVD (df1)')
axes[1, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))
# 4. TSNE
#df1_sample_tsne_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df1_sample_tsne_2, x='Component 1', y='Component 2', hue='dbscan', style='dbscan',
               palette='tab20', ax=axes[1, 1])
axes[1, 1].set_title('TSNE (df1)')
axes[1, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.suptitle(f"DBSCAN - {tsne.perplexity}")
fig.subplots_adjust(top=0.95)
plt.legend(loc=1, bbox_to_anchor=(1.2, 1))
plt.show()

#ANÀLISI 
#MEDIES DE LES FEATURES PER CADA CLUSTER
df1_hdbscan1 = pd.DataFrame(data=data1_scal, columns=df1.columns)
df1_hdbscan1["label"] = hdbscan1

tmp = df1_hdbscan1.groupby(by=["label"]).mean()

plt.figure(figsize=(35, 15))
sns.heatmap(tmp, cmap="Spectral", annot=True)

#DISTRIBUCIÓ DE LES FEATURES PER CADA CLUSTER
fig, axes = plt.subplots(len(df1_hdbscan1.columns), figsize=(30, 200))

for i, c in enumerate(df1_hdbscan1.columns): 
    sns.kdeplot(df1_hdbscan1, x=c, hue="label", ax=axes[i])

plt.show()

#BARPLOTS DE LES FEATURES PER CADA CLUSTER
fig, axes = plt.subplots(len(df1_hdbscan1.columns), figsize=(30, 200))

for i, c in enumerate(df1_hdbscan1.columns): 
    sns.barplot(df1_hdbscan1, y=c, x="label", ax=axes[i])

plt.show()


#BOXPLOTS DE LES FEATURES PER CADA CLUSTER
fig, axes = plt.subplots(len(df1_hdbscan1.columns), figsize=(30, 200))

for i, c in enumerate(df1_hdbscan1.columns): 
    sns.boxplot(df1_hdbscan1, x="label", y=c, ax=axes[i])

plt.show()

#FEATURE IMPORTANCE AMB RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier


for i in range(-1, len(set(hdbscan1))-1):
    df1_hdbscan1["cluster_"+str(i)] = (df1_hdbscan1["label"] == i)

    clf = RandomForestClassifier(random_state=1)
    clf.fit(df1_hdbscan1.iloc[:, 0:40].values, df1_hdbscan1["cluster_"+str(i)].values)

    importance = clf.feature_importances_
    #plt.bar(range(num.shape[1]), importance)
    plt.bar(range(40), importance)
    plt.xticks(range(40), df1_hdbscan1.columns[0:40], rotation=90)
    plt.title("Feature Importance in Random Forest - clúster " + str(i))
    plt.show()


#EXEMPLES RANDOM SONGS
cluster_labels = sorted([label for label in set(hdbscan1) if label != -1]) #totes les labels del model
sizes = {label: int((hdbscan1 == label).sum()) for label in cluster_labels} #quanitat de cancos en cada label

random_index = np.random.RandomState(42).randint(len(data1_scal)) #index de canço random
random_label = hdbscan1[random_index] #label de canço random

cluster_index = np.where(hdbscan1 == random_label)[0] #index de cancons amb label de canço random 

recomancio = 2
random_recomancio_index = np.random.RandomState(42).choice(cluster_index, size=recomancio, replace=False) #index de 

#print("Nom canço:", canciones.loc[:, "track_name"][random_index].values)
#print("Nom artista:", canciones.loc[:, "artist_name"][random_index].values)
print("Nom canço:", canciones["track_name"][random_index])
print("Nom artista:", canciones["artist_name"][random_index])
print("Label:", hdbscan1[random_index])
print("Cançons disponibles:", sizes[random_label])

for i in range(recomancio): 
    print("RECOMANCIO", i)
    print("Nom canço:", canciones["track_name"][random_recomancio_index[i]])
    print("Nom artista:", canciones["artist_name"][random_recomancio_index[i]])
    print("Label:", hdbscan1[random_recomancio_index[i]])