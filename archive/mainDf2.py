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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
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
df = pd.read_csv('SpotifyFeatures.csv')
#print(df.head()) #Mostra 5 rows
print(df.info()) #Mostra el tipus de columns: df.dtypes
##C)Preparar dades
#1) Tractar duplicats
#Quantitat duplicats
print("Quantitat de dupplicats:", df.duplicated(subset="track_id", keep="first").sum())
#print(df[df.duplicated(subset="track_id", keep=False)]) Veure files dels duplicats

#Treure duplicats (files)
df.drop_duplicates(subset="track_id", keep="first", inplace=True)
print("Quantitat de dupplicats després:", df.duplicated(subset="track_id", keep="first").sum())
print("New shape sense duplicats:", df.shape)

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
df.drop(['artist_name','track_name','track_id', 'genre', 'popularity'], axis=1, inplace=True) #Elimina les columnes modificant df og
print(f"Nova shape sense artist_name, track_name, track_id: {df.shape}") #Mostra rowsxcolumns

num_col = df.select_dtypes(exclude="object").columns
cat_col = df.select_dtypes(include="object").columns

#4)Codificar columnes no numèriques
#Diccionari
df6 = df.copy(deep=True)
freq = df6["key"].value_counts(normalize=True)
df6["key"] = df6["key"].map(freq)
#Codificar columna mode 
freq = df["mode"].value_counts(normalize=True)
df6["mode"] = df6["mode"].map(freq)
#Codificar columna time_signature
freq = df6["time_signature"].value_counts(normalize=True)
df6["time_signature"] = df6["time_signature"].map(freq)

print("Nova shape df6:", df6.shape)
print("Noves columnas:", df6.columns)

#5) Mirar histograma
df6.hist(figsize=(30, 30)), plt.show()

#6) Mirar correlació (pearson)
corr_matrix3 = df6[num_col].corr()
plt.figure(figsize=(12, 12))
ax = sns.heatmap(corr_matrix3, annot=True, cmap='coolwarm')
plt.show()
#7) Escalar columnes
scaler = MinMaxScaler()
tmp = df6.copy()
tmp["duration_ms"] = scaler.fit_transform(df6[["duration_ms"]])
tmp["loudness"] = scaler.fit_transform(df6[["loudness"]])
tmp["tempo"] = scaler.fit_transform(df6[["tempo"]])
#tmp["key"] = scaler.fit_transform(df6[["key"]])
#tmp["time_signature"] = scaler.fit_transform(df6[["time_signature"]])

data6_scal = tmp.values
#df6_scal = pd.DataFrame(data=data6_scal, columns=df6.columns)

#ESTABLECER MODELO
#DF6 KMEANS
#1) Calcular k millor
model = KMeans(n_init=10, random_state=0)
visualizer = KElbowVisualizer(model, k=(2,30), timings=False, metric='distortion') #default distance_metric: euclidean
visualizer.fit(data6_scal)       
#visualizer.show() #Quitar en py
#visualizer.show(outpath="kmeans.png") #Guardar gráfica
k = visualizer.elbow_value_ #millor k
#2) Entrenar i predir amb KMEANS
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
#kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0)
kmeans1 = kmeans.fit_predict(data6_scal) #Mateix que: kmeans.labels__
#data_scal["cluster"] = pred

#DF6 DBSCAN
#1) Calcular eps
min_samples = 2 * data6_scal.shape[1]
nn = NearestNeighbors(n_neighbors=min_samples).fit(data6_scal) #default algorithm: auto; most appropiate, default metric: minkowski
dists, _ = nn.kneighbors(data6_scal)
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
min_samples = 2 * data6_scal.shape[1]
dbscan = DBSCAN(eps=0.37, min_samples=min_samples, n_jobs=-1,) #default algorithm: auto of NearestNeighbours; most appropiate
dbscan1 = dbscan.fit_predict(data6_scal)
#print(calinski_harabasz_score(data6_scal, dbscan1), davies_bouldin_score(data6_scal, dbscan1))

#DF6 HDBSCAN
#1) Entrenar i predir amb HDBSCAN
min_samples = 2 * data6_scal.shape[1]
hdbscan = HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples, n_jobs=-1, store_centers="centroid") #default metric: euclidean
hdbscan1 = hdbscan.fit_predict(data6_scal)

#DF6 GMM
#1) Trobar n_components
n_components_range = range(18, 30)
bics = []
aics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, n_init=10, random_state=0)
    gmm.fit(data6_scal)
    bics.append(gmm.bic(data6_scal))
    aics.append(gmm.aic(data6_scal))

plt.figure(figsize=(12, 6))
plt.plot(n_components_range, bics, label='BIC')
plt.plot(n_components_range, aics, label='AIC')
plt.xlabel('N_components')
plt.ylabel('BIC / AIC')
plt.legend(loc=2)
plt.title('BIC and AIC segons components')
plt.savefig("gm.png")
plt.show()
#2) Entrenar i predir amb GM
gm = GaussianMixture(n_components=25, n_init=10, random_state=0) #default covariance_type: full 
gm1 = gm.fit_predict(data6_scal)

#EVALUAR MODELS
#KMEANS
print(calinski_harabasz_score(data6_scal, kmeans1), davies_bouldin_score(data6_scal, kmeans1))
#DBSCAN
print(calinski_harabasz_score(data6_scal, dbscan1), davies_bouldin_score(data6_scal, dbscan1))
#HDBSCAN
print(calinski_harabasz_score(data6_scal, hdbscan1), davies_bouldin_score(data6_scal, hdbscan1))
#GaussianMixture
print(calinski_harabasz_score(data6_scal, gm1), davies_bouldin_score(data6_scal, gm1))

#VISUALITZAR CLUSTERS 
# Crear sampled dataset per proves més ràpides
# Samplearem 15% de les dades per a proves
sample_size = int(len(df) * 0.15)
#sample_size = int(len(df) * 0.15 / len(set(hdbscan1)))

#df6_kmeans1 = pd.DataFrame(data=data6_scal, columns=df6.columns)
#df6_kmeans1["label"] = hdbscan1
#sample_indices = df6_kmeans1.groupby("label").apply(lambda x: x.sample(sample_size, random_state=0) if len(x) >= sample_size else x, include_groups=False).droplevel("label").index
sample_indices = np.random.RandomState(42).choice(len(df), sample_size, replace=False)

df6_sample = data6_scal[sample_indices]
#pred= pred[sample_indices]
print("Mida orignal:", df.shape)
print("Mida 20%:", df6_sample.shape)
# 1. PCA Estàndard
pca_sample = PCA(n_components=2, random_state=42)
df6_sample_pca = pca_sample.fit_transform(df6_sample)

df6_sample_pca_2 = pd.DataFrame(data=df6_sample_pca, columns=['Component 1', 'Component 2'])
df6_sample_pca_2["kmeans"] = kmeans1[sample_indices]
df6_sample_pca_2["dbscan"] = dbscan1[sample_indices]
df6_sample_pca_2["hdbscan"] = hdbscan1[sample_indices]

# 2. KernelPCA RBF
kpca_sample = KernelPCA(n_components=2, kernel='rbf', random_state=42, n_jobs=-1)
df6_sample_kpca = kpca_sample.fit_transform(df6_sample)

df6_sample_kpca_2 = pd.DataFrame(data=df6_sample_kpca, columns=['Component 1', 'Component 2'])
df6_sample_kpca_2["kmeans"] = kmeans1[sample_indices]
df6_sample_kpca_2["dbscan"] = dbscan1[sample_indices]
df6_sample_kpca_2["hdbscan"] = hdbscan1[sample_indices]

# 3. TruncatedSVD
tsvd_sample = TruncatedSVD(n_components=2, random_state=42)
df6_sample_tsvd = tsvd_sample.fit_transform(df6_sample)

df6_sample_tsvd_2 = pd.DataFrame(data=df6_sample_tsvd, columns=['Component 1', 'Component 2'])
df6_sample_tsvd_2["kmeans"] = kmeans1[sample_indices]
df6_sample_tsvd_2["dbscan"] = dbscan1[sample_indices]
df6_sample_tsvd_2["hdbscan"] = hdbscan1[sample_indices]

# 4. TSNE
tsne = TSNE(n_components=2, perplexity=150, learning_rate='auto', random_state=42)
df6_sample_tsne = tsne.fit_transform(df6_sample)
df6_sample_tsne_2 = pd.DataFrame(data=df6_sample_tsne, columns=['Component 1', 'Component 2'])
df6_sample_tsne_2["kmeans"] = kmeans1[sample_indices]
df6_sample_tsne_2["dbscan"] = dbscan1[sample_indices]
df6_sample_tsne_2["hdbscan"] = hdbscan1[sample_indices]
pca_sample.explained_variance_ratio_.cumsum(), tsvd_sample.explained_variance_ratio_.cumsum(), tsne.kl_divergence_

pca_sample.explained_variance_ratio_.cumsum(), tsvd_sample.explained_variance_ratio_.cumsum(), tsne.kl_divergence_
#KMEANS
fig, axes = plt.subplots(2, 2, figsize=(26, 24))

# 1. PCA Estàndard
#df6_sample_pca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_pca_2, x='Component 1', y='Component 2', hue='kmeans', style='kmeans',
               palette='tab20', ax=axes[0, 0])
axes[0, 0].set_title('PCA Estàndard (df6)')
axes[0, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))

# 2. KernelPCA RBF
#df6_sample_kpca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_kpca_2, x='Component 1', y='Component 2', hue='kmeans',  style='kmeans',
               palette='tab20', ax=axes[0, 1])
axes[0, 1].set_title('KernelPCA RBF (df6)')
axes[0, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

# 3. TruncatedSVD
#df6_sample_tsvd_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_tsvd_2, x='Component 1', y='Component 2', hue='kmeans',  style='kmeans',
               palette='tab20', ax=axes[1, 0])
axes[1, 0].set_title('TruncatedSVD (df6)')
axes[1, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))

# 4. TSNE
#df6_sample_tsne_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_tsne_2, x='Component 1', y='Component 2', hue='kmeans', style='kmeans',
               palette='tab20', ax=axes[1, 1])
axes[1, 1].set_title('TSNE (df6)')
axes[1, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.suptitle(f"KMEANS - {tsne.perplexity}")
fig.subplots_adjust(top=0.95)
plt.legend(loc=1, bbox_to_anchor=(1.2, 1))
plt.show()

#DBSCAN
fig, axes = plt.subplots(2, 2, figsize=(26, 24))

# 1. PCA Estàndard
#df6_sample_pca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_pca_2, x='Component 1', y='Component 2', hue='dbscan', style='dbscan',
               palette='tab20', ax=axes[0, 0])
axes[0, 0].set_title('PCA Estàndard (df6)')
axes[0, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))

# 2. KernelPCA RBF
#df6_sample_kpca_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_kpca_2, x='Component 1', y='Component 2', hue='dbscan',  style='dbscan',
               palette='tab20', ax=axes[0, 1])
axes[0, 1].set_title('KernelPCA RBF (df6)')
axes[0, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

# 3. TruncatedSVD
#df6_sample_tsvd_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_tsvd_2, x='Component 1', y='Component 2', hue='dbscan',  style='dbscan',
               palette='tab20', ax=axes[1, 0])
axes[1, 0].set_title('TruncatedSVD (df6)')
axes[1, 0].legend(loc=2, bbox_to_anchor=(-0.3, 1))

# 4. TSNE
#df6_sample_tsne_2["Cluster"] = hdbscan1[sample_indices]
sns.scatterplot(data=df6_sample_tsne_2, x='Component 1', y='Component 2', hue='dbscan', style='dbscan',
               palette='tab20', ax=axes[1, 1])
axes[1, 1].set_title('TSNE (df6)')
axes[1, 1].legend(loc=1, bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.suptitle(f"DBSCAN - {tsne.perplexity}")
fig.subplots_adjust(top=0.95)
plt.legend(loc=1, bbox_to_anchor=(1.2, 1))
plt.show()

#ANALISIS
#MEDIAS CLUSTERS KMEANS
df6_kmeans1 = pd.DataFrame(data=data6_scal, columns=df6.columns)
df6_kmeans1["label"] = kmeans1

tmp = df6_kmeans1.groupby(by=["label"]).mean()

plt.figure(figsize=(35, 15))
sns.heatmap(tmp, cmap="Spectral", annot=True)

#DISTRIBUCIONS FEATURES PER CLUSTERS KMEANS
fig, axes = plt.subplots(len(df6_kmeans1.columns), figsize=(30, 200))

for i, c in enumerate(df6_kmeans1.columns): 
    sns.kdeplot(df6_kmeans1, x=c, hue="label", ax=axes[i])

plt.show()
#BARPLOTS FEATURES PER CLUSTERS KMEANS
fig, axes = plt.subplots(len(df6_kmeans1.columns), figsize=(30, 200))

for i, c in enumerate(df6_kmeans1.columns): 
    sns.barplot(df6_kmeans1, y=c, x="label", ax=axes[i])

plt.show()

#FEATURE IMPORTANCE AMB RANDOM FOREST PER CLUSTERS KMEANS
from sklearn.ensemble import RandomForestClassifier
for i in range(len(set(kmeans1))):
    df6_kmeans1["cluster_"+str(i)] = (df6_kmeans1["label"] == i)

    clf = RandomForestClassifier(random_state=1)
    clf.fit(df6_kmeans1.iloc[:, 0:13].values, df6_kmeans1["cluster_"+str(i)].values)

    importance = clf.feature_importances_
    #plt.bar(range(num.shape[1]), importance)
    plt.bar(range(13), importance)
    plt.xticks(range(13), df6_kmeans1.columns[0:13], rotation=90)
    plt.title("Feature Importance in Random Forest - clúster " + str(i))
    plt.show()

#pd.DataFrame(data=scaler.inverse_transform(tmp), columns=df.columns)
tmp1 = pd.DataFrame(data=tmp, columns=df.columns)
scaler.fit(df6[["duration_ms"]])
tmp1["duration_ms"] = scaler.inverse_transform(tmp[["duration_ms"]])

scaler.fit(df6[["loudness"]])
tmp1["loudness"] = scaler.inverse_transform(tmp[["loudness"]])

scaler.fit(df6[["tempo"]])
tmp1["tempo"] = scaler.inverse_transform(tmp[["tempo"]])

tmp1

#RANDOM SONG
cluster_labels = sorted([label for label in set(kmeans1) if label != -1]) #totes les labels del model
sizes = {label: int((kmeans1 == label).sum()) for label in cluster_labels} #quanitat de cancos en cada label

random_index = np.random.RandomState(42).randint(len(data6_scal)) #index de canço random
random_label = kmeans1[random_index] #label de canço random

cluster_index = np.where(kmeans1 == random_label)[0] #index de cancons amb label de canço random 

recomancio = 2
random_recomancio_index = np.random.RandomState(42).choice(cluster_index, size=recomancio, replace=False) #index de 

#print("Nom canço:", canciones.loc[:, "track_name"][random_index].values)
#print("Nom artista:", canciones.loc[:, "artist_name"][random_index].values)
print("Nom canço:", canciones["track_name"][random_index])
print("Nom artista:", canciones["artist_name"][random_index])
print("Label:", kmeans1[random_index])
print("Cançons disponibles:", sizes[random_label])
print("Info:", df.iloc[random_index])

for i in range(recomancio): 
    print("RECOMANCIO", i)
    print("Nom canço:", canciones["track_name"][random_recomancio_index[i]])
    print("Nom artista:", canciones["artist_name"][random_recomancio_index[i]])
    print("Label:", kmeans1[random_recomancio_index[i]])
    print("Info:", df.iloc[random_recomancio_index[i]])