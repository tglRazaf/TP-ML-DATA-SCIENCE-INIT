import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
df = pd.read_csv("assets/Mall_Customers.csv")


# 2. Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# 3. Standardisation des variables numériques
scaler = StandardScaler()
df_scaled = df.copy()

# Colonnes à standardiser
cols_to_scale = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 4. Aperçu des données standardisées
print(df_scaled.head())
