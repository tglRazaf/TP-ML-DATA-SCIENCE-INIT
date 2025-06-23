import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from age_group_feature import df_features

# 2. Appliquer la PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_features)

# 3. Créer un DataFrame pour affichage
df_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])

# 4. Affichage en 2D
plt.figure(figsize=(8,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.7)
plt.title("Projection des données après PCA")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.tight_layout()
plt.savefig("pca_2d_projection.png")  # ou plt.show() si en notebook
plt.close()
