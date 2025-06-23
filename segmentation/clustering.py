from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pca_application import df_features

inertias = []
silhouette_scores = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_features)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_features, kmeans.labels_))

# Méthode du coude (Elbow)
plt.figure(figsize=(6, 4))
plt.plot(k_range, inertias, 'bo-', marker='o')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie (Within-Cluster SSE)")
plt.title("Méthode du coude")
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.close()

# Silhouette Score
plt.figure(figsize=(6, 4))
plt.plot(k_range, silhouette_scores, 'go-', marker='o')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score de silhouette")
plt.title("Score de silhouette selon k")
plt.tight_layout()
plt.savefig("silhouette_score.png")
plt.close()
