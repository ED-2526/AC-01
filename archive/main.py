# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix

df = pd.read_csv("archive/SpotifyFeatures.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.columns)

df['genre'] = df['genre'].astype('category')
df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')
df['valence'] = df['valence'].astype('category')

df_clean = df.drop(columns=['track_id', 'time_signature', 'artist_name', 'track_name'])

print(df_clean.dtypes)

np.random.seed(50)

intrain = np.random.choice(df_clean.index, size=int(len(df_clean) * 0.008593834), replace=False)

df_sample = df_clean.loc[intrain]

df_sample.shape

df_clean.isna().sum()

df_num = df_sample.select_dtypes(include='number')

scaler = StandardScaler()

df_scale = scaler.fit_transform(df_num)

df_scale = pd.DataFrame(df_scale, columns=df_num.columns)

df_scale.plot(kind='box', subplots=True, layout=(2, 11), sharex=False, sharey=False, figsize=(20, 10))
plt.show()

df_scale.hist(bins=30, figsize=(12, 10))
plt.show()

scatter_matrix(df_scale, figsize=(15, 15))
plt.show()

correlation_matrix = df_scale.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriks Korelasi Antar Variabel")
plt.show()