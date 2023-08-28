import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


data = pd.read_csv('flats_for_clustering.tsv', sep='\t')

# replace Nan


data['Piętro'] = data['Piętro'].replace({'parter' : '0', 'poddasze' : '4','niski parter' : '0.5'})
data['Piętro'] = pd.to_numeric(data['Piętro'])
data = data.dropna(axis=0)
# print(data)

# k-means start

scaled = StandardScaler().fit_transform(data)
model = KMeans(n_clusters=5)
cluster_labels = model.fit_predict(scaled)
data['Number_of_cluster'] = cluster_labels
# print(data)

# PCA

data = data.drop('Number_of_cluster', axis=1)
scaled = StandardScaler().fit_transform(data)
pca_model = PCA(n_components=2)
pca_results = pca_model.fit_transform(scaled)


# wykres

plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()











