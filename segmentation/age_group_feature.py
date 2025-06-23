from data_preparation import df_scaled, df

# Remove customer ID (Inused for analysis)
df_features = df_scaled.drop(columns=["CustomerID"], errors='ignore')

# Encode gender
df_features["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Creation new feature for age group
df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 25, 35, 50, 70],
                        labels=["18-25", "26-35", "36-50", "51-70"])

df = pd.get_dummies(df, columns=["AgeGroup"], drop_first=True)
