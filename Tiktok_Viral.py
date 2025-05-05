import pandas as pd
import ast
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 1. Charger le dataset existant

df = pd.read_csv("/Users/giulianodarwish/Documents/ML_PROJECT/trending_converted.csv")

# 🔥 AJOUT → charger le deuxième dataset
df_meta = pd.read_csv("/Users/giulianodarwish/Documents/ML_PROJECT/meta_data.csv")

# 🔥 AJOUT → concaténer verticalement
df = pd.concat([df, df_meta], ignore_index=True)

print(f"✅ Datasets concaténés : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# (tout le reste de ton code continue normalement ici)



# 2. Créer la variable cible "viral" (1 si playCount > 100 000, sinon 0)
threshold = 100_000
df["viral"] = (df["playCount"] > threshold).astype(int)

# 3. Définir la liste des colonnes “pré-publication” à conserver + la cible
columns_to_keep = [
    # Texte et métadonnées
    "text",
    "hashtags",
    "mentions",
    "createTime",

    # Auteur
    "authorMeta.id",
    "authorMeta.secUid",
    "authorMeta.name",
    "authorMeta.nickName",
    "authorMeta.verified",

    # Musique
    "musicMeta.musicId",
    "musicMeta.musicName",
    "musicMeta.musicAuthor",
    "musicMeta.musicOriginal",

    # Vidéo
    "videoMeta.duration",

    # Cible
    "viral"
]

# 4. Filtrer le DataFrame
df = df[columns_to_keep].copy()

# 5. (Optionnel) Convertir createTime en datetime
df["createTime"] = pd.to_datetime(df["createTime"], unit="s")

# 6. Sauvegarder le résultat
output_path = "/Users/giulianodarwish/Documents/ML_PROJECT/Trending_pre_publication_with_target.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Dataset filtré + cible créé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
print(f"Fichier sauvegardé sous : {output_path}")

# FEATURE ENGINEERING
# 1. Nettoyage initial
#   - supprimer les lignes où la description est manquante ou vide
df_pre = df.dropna(subset=["text", "createTime"])
df_pre = df_pre[df_pre["text"].str.strip() != ""]

# 2. Convertir createTime en datetime
df_pre["createTime"] = pd.to_datetime(df_pre["createTime"], unit="s")

# 3. Feature Engineering
#   3.1 description_length
df_pre["description_length"] = df_pre["text"].str.len()

#   3.2 hashtags → liste, n_hashtags & has_hashtags
df_pre["hashtags_list"] = df_pre["hashtags"].apply(ast.literal_eval)
df_pre["n_hashtags"] = df_pre["hashtags_list"].apply(len)
df_pre["has_hashtags"] = (df_pre["n_hashtags"] > 0).astype(int)

#   3.3 author_verified et music_is_original en entier
df_pre["author_verified"]   = df_pre["authorMeta.verified"].astype(int)
df_pre["music_is_original"] = df_pre["musicMeta.musicOriginal"].astype(int)

#   3.4 heure et jour de publication
df_pre["hour_posted"] = df_pre["createTime"].dt.hour
df_pre["day_of_week"] = df_pre["createTime"].dt.dayofweek

# 4. Encodage de musicMeta.musicAuthor (on garde les 10 plus fréquents, le reste = 'other')
top_authors = df_pre["musicMeta.musicAuthor"].value_counts().nlargest(10).index
df_pre["musicAuthor_top"] = df_pre["musicMeta.musicAuthor"].where(
    df_pre["musicMeta.musicAuthor"].isin(top_authors),
    "other"
)
df_pre = pd.get_dummies(df_pre, columns=["musicAuthor_top"], prefix="musicAuthor")

# 5. Nettoyage final : supprimer colonnes brutes devenues inutiles
cols_to_drop = [
    "hashtags", "hashtags_list", "mentions",
    "authorMeta.verified", "musicMeta.musicOriginal", "musicMeta.musicAuthor"
]
df_pre.drop(columns=cols_to_drop, inplace=True)
cols = [c for c in df_pre.columns if c != "viral"] + ["viral"]
df_pre = df_pre[cols]

# 6. Sauvegarder le dataframe prêt pour modélisation
output_path = "/Users/giulianodarwish/Documents/ML_PROJECT/Trending_model_ready.csv"
df_pre.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Feature engineering et nettoyage terminés : {df_pre.shape[0]} lignes, {df_pre.shape[1]} colonnes.")
print(f"Fichier prêt pour modélisation : {output_path}")


numeric_df = df_pre.select_dtypes(include=[np.number])
X = numeric_df.drop(columns=["viral"])
y = numeric_df["viral"]

# 3. Matrice de corrélation
corr_matrix = X.corr()
print("=== Matrice de corrélation ===")
print(corr_matrix.round(2))

# 4. Variance Inflation Factor (VIF)
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
}).sort_values("VIF", ascending=False)
print("\n=== VIF par variable ===")
print(vif)

# 5. Paires trop corrélées (|ρ| > 0.7)
high_corr = []
for i, feat_i in enumerate(corr_matrix.columns):
    for j, feat_j in enumerate(corr_matrix.columns[:i]):
        rho = corr_matrix.iloc[i, j]
        if abs(rho) > 0.7:
            high_corr.append((feat_i, feat_j, rho))
print("\n=== Paires corrélées > 0.7 ===")
for a, b, rho in high_corr:
    print(f"{a} ↔ {b} : ρ = {rho:.2f}")

# 6. Standardisation + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)
explained = pd.Series(pca.explained_variance_ratio_,
                      index=[f"PC{i+1}" for i in range(len(X.columns))])
print("\n=== Variance expliquée par composante PCA ===")
print(explained.head(10).round(3))

# 7. Régression pénalisée (Logistic Regression)
# 7.1 L2 (Ridge)
clf_l2 = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
auc_l2 = cross_val_score(clf_l2, X_scaled, y, cv=5, scoring="roc_auc").mean()
clf_l2.fit(X_scaled, y)
coef_l2 = pd.Series(clf_l2.coef_[0], index=X.columns).abs().sort_values(ascending=False)

# 7.2 L1 (Lasso)
clf_l1 = LogisticRegression(penalty="l1", solver="saga", max_iter=1000)
auc_l1 = cross_val_score(clf_l1, X_scaled, y, cv=5, scoring="roc_auc").mean()
clf_l1.fit(X_scaled, y)
coef_l1 = pd.Series(clf_l1.coef_[0], index=X.columns).abs().sort_values(ascending=False)

print(f"\n=== Performance (ROC AUC) ===\nL2 : {auc_l2:.3f}\nL1 : {auc_l1:.3f}")

print("\n=== Top 10 coefficients L2 (abs) ===")
print(coef_l2.head(10))

print("\n=== Top 10 coefficients L1 (abs) ===")
print(coef_l1.head(10))
