import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import df_scaled

# 1. Histogrammes pour chaque variable
df_scaled.hist(bins=20, figsize=(12, 6), grid=False)
plt.suptitle("Histogrammes des variables", fontsize=16)
plt.show()

# 2. Boxplots pour visualiser les outliers
plt.figure(figsize=(12, 5))
for i, col in enumerate(["Age", "Annual Income (k$)", "Spending Score (1-100)"]):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df_scaled[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()

# 3. Matrice de corrélation
plt.figure(figsize=(6, 4))
sns.heatmap(df_scaled[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# 4. Scatterplot matrix (pairplot)
sns.pairplot(df_scaled[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
plt.suptitle("Scatterplot matrix", y=1.02)
plt.show()
