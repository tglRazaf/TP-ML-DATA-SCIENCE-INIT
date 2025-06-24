# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 2. Chargement du fichier CSV
df = pd.read_csv('assets/diabetes.csv')

# 3. Nettoyage des données
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# Remplacement des valeurs manquantes par la médiane
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].fillna(df[cols_with_invalid_zeros].median())

# 4. Standardisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))
df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
df_scaled['Outcome'] = df['Outcome']

# 5. Analyse exploratoire (EDA)
sns.countplot(x='Outcome', data=df_scaled)
plt.title('Répartition des classes (0 = Non diabétique, 1 = Diabétique)')
plt.xlabel('Classe')
plt.ylabel('Nombre d\'échantillons')
plt.show()

print("Statistiques descriptives par classe :")
print(df_scaled.groupby('Outcome').mean())

# 6. Split Train / Validation / Test
X = df_scaled.drop('Outcome', axis=1)
y = df_scaled['Outcome']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# 7. Modélisation : Régression Logistique
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Évaluation sur Validation Set
y_val_pred = model.predict(X_val)
print("\nÉvaluation sur le set de validation :")
print("Accuracy :", accuracy_score(y_val, y_val_pred))
print("Precision :", precision_score(y_val, y_val_pred))
print("Recall :", recall_score(y_val, y_val_pred))
print("F1-score :", f1_score(y_val, y_val_pred))

# 9. Évaluation sur Test Set
y_test_pred = model.predict(X_test)
print("\nMatrice de confusion (Test set) :")
print(confusion_matrix(y_test, y_test_pred))

print("\nRapport de classification (Test set) :")
print(classification_report(y_test, y_test_pred))

# 10. Analyse de l’importance des features
importance = pd.Series(abs(model.coef_[0]), index=X.columns).sort_values(ascending=False)
print("\nImportance des features :")
print(importance)

# 11. Conclusion
print("\nConclusions :")
print("- Le modèle de régression logistique donne une base simple mais efficace.")
print("- Les features les plus importantes sont généralement le glucose, l’IMC, et l’âge.")
print("- Pour améliorer les performances : considérer d'autres modèles, équilibrer davantage les classes, ou explorer l'interaction entre variables.")
print("- Il est recommandé d'assurer la qualité des données collectées pour les futures prédictions (éviter les 0 cliniquement non valides).")
