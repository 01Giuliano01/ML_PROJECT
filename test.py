import pandas as pd
import ast
import numpy as np
import difflib

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 1. Charger les deux datasets
# Remplacez par vos chemins
df1 = pd.read_csv("/Users/giulianodarwish/Documents/ML_PROJECT/tiktok_dataset.csv")
df2 = pd.read_csv("/Users/giulianodarwish/Documents/ML_PROJECT/meta_data.csv")

# 2. Détecter automatiquement les colonnes similaires (mêmes variables, noms différents)
cols1 = df1.columns.tolist()
cols2 = df2.columns.tolist()

# Seuil de similarité (0.6 à ajuster)
cutoff = 0.6
mapping = {}
scores = {}
for col2 in cols2:
    # chercher la colonne la plus proche dans cols1
    matches = difflib.get_close_matches(col2, cols1, n=1, cutoff=cutoff)
    if matches:
        best = matches[0]
        # calculer ratio exact
        ratio = difflib.SequenceMatcher(None, col2, best).ratio()
        mapping[col2] = best
        scores[col2] = ratio

# Afficher les correspondances détectées
print("Correspondances détectées entre df2 -> df1:")
for k, v in mapping.items():
    print(f"  {k} -> {v} (similarité: {scores[k]:.2f})")

# 3. Renommer df2 selon mapping
df2_renamed = df2.rename(columns=mapping)

# 4. Concaténation verticale
# Les colonnes renommées dans df2_renamed coïncident avec celles de df1
df = pd.concat([df1, df2_renamed], ignore_index=True)
print(f"✅ Datasets concaténés : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# 5. Unifier le compte de vues (playCount)
# df1: 'video_view_count', df2: peut-être renommé en 'video_view_count' ou 'plays'
df['playCount'] = df.get('video_view_count', np.nan).combine_first(df.get('plays', np.nan))
# Remplacer NaN
df['playCount'] = df['playCount'].fillna(0)

# 6. Créer la variable cible 'viral'
threshold = 100_000
df['viral'] = (df['playCount'] > threshold).astype(int)

# 7. Sélection des colonnes pré-publication + cible (adapter selon vos variables)
columns_to_keep = [
    # Exemple: remplacer par vos colonnes alignées
    'video_transcription_text', 'hashtags', 'timestamp',
    'verified_status', 'author_followers', 'author_likes', 'author_videos',
    'music_id_hash', 'music_name_hash', 'music_author_hash', 'video_duration_sec',
    'viral'
]

# Ne garder que les colonnes existantes
columns_to_keep = [col for col in columns_to_keep if col in df.columns]
df = df[columns_to_keep].copy()

# 8. Convertir timestamp en datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

# 9. Sauvegarde du dataset filtré
output_path1 = "/Users/giulianodarwish/Documents/ML_PROJECT/Trending_pre_publication_with_target.csv"
df.to_csv(output_path1, index=False, encoding="utf-8")
print(f"✅ Dataset filtré + cible créé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# --- FEATURE ENGINEERING ---
# 10. Nettoyage initial
df_pre = df.dropna(subset=[col for col in ['video_transcription_text','timestamp'] if col in df.columns]).copy()
df_pre = df_pre[df_pre['video_transcription_text'].str.strip() != ''] if 'video_transcription_text' in df_pre.columns else df_pre

# 11. description_length
if 'video_transcription_text' in df_pre.columns:
    df_pre['description_length'] = df_pre['video_transcription_text'].str.len()

# 12. hashtags → liste, n_hashtags, has_hashtags
if 'hashtags' in df_pre.columns:
    try:
        df_pre['hashtags_list'] = df_pre['hashtags'].apply(ast.literal_eval)
        df_pre['n_hashtags'] = df_pre['hashtags_list'].apply(len)
        df_pre['has_hashtags'] = (df_pre['n_hashtags'] > 0).astype(int)
    except:
        df_pre['n_hashtags'] = 0; df_pre['has_hashtags'] = 0

# 13. author_verified
if 'verified_status' in df_pre.columns:
    df_pre['author_verified'] = df_pre['verified_status'].astype(int)

# 14. heure/jour
date_col = 'timestamp' if 'timestamp' in df_pre.columns else None
if date_col:
    df_pre['hour_posted'] = df_pre[date_col].dt.hour
    df_pre['day_of_week'] = df_pre[date_col].dt.dayofweek

# 15. Nettoyage des colonnes brutes inutiles
drop_cols = ['hashtags', 'verified_status', 'plays', 'video_view_count']
drop_cols = [c for c in drop_cols if c in df_pre.columns]
if 'hashtags_list' in df_pre.columns: drop_cols.append('hashtags_list')
df_pre.drop(columns=drop_cols, inplace=True, errors='ignore')
# Reorder pour mettre viral en fin
cols = [c for c in df_pre.columns if c!='viral'] + ['viral']
df_pre = df_pre[cols]

# 16. Sauvegarde du dataset prêt pour modélisation
output_path2 = "/Users/giulianodarwish/Documents/ML_PROJECT/Trending_model_ready.csv"
df_pre.to_csv(output_path2, index=False, encoding="utf-8")
print(f"✅ Feature engineering terminé : {df_pre.shape[0]} lignes, {df_pre.shape[1]} colonnes.")

# --- ANALYSE ET MODÉLISATION ---
numeric_df = df_pre.select_dtypes(include=[np.number])
X = numeric_df.drop(columns=['viral'], errors='ignore')
y = numeric_df['viral'] if 'viral' in numeric_df.columns else None

# Corrélation & VIF
if not X.empty and y is not None:
    corr = X.corr(); print('=== Corrélation ==='); print(corr.round(2))
    vif = pd.DataFrame({'feature':X.columns,'VIF':[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]}).sort_values('VIF',ascending=False)
    print('=== VIF ==='); print(vif)

    # PCA
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    pca = PCA().fit(Xs); evr = pd.Series(pca.explained_variance_ratio_, index=[f'PC{i+1}' for i in range(X.shape[1])])
    print('=== PCA variance ratio ==='); print(evr.head(10).round(3))

    # Modèles
    clf_l2 = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)
    auc_l2 = cross_val_score(clf_l2,Xs,y,cv=5,scoring='roc_auc').mean()
    clf_l1 = LogisticRegression(penalty='l1',solver='saga',max_iter=1000)
    auc_l1 = cross_val_score(clf_l1,Xs,y,cv=5,scoring='roc_auc').mean()
    print(f'ROC AUC → L2: {auc_l2:.3f}, L1: {auc_l1:.3f}')
