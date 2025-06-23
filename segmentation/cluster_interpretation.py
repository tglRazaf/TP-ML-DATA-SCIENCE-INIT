from clustering import df_features
from sklearn.cluster import KMeans
from pca_application import df_pca
# cluster_profiles = df_features.groupby("Cluster").mean()
# print(cluster_profiles)


# Appliquer le clustering final
k_final = 4
kmeans = KMeans(n_clusters=k_final, random_state=42)
cluster_labels = kmeans.fit_predict(df_features)

# Ajouter les labels à vos données PCA et originales
df_pca["Cluster"] = cluster_labels
df_features["Cluster"] = cluster_labels

# Profil moyen par cluster
cluster_profiles = df_features.groupby("Cluster").mean()
print(cluster_profiles)
