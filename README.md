# TikTok Virality Prediction — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

> End-to-end machine learning project to **predict whether a TikTok video will go viral** (>100,000 views) using only **pre-publication features** — metadata available before posting, with no engagement data.

---

## Problem Statement

Can we predict TikTok virality **before** a video is published? This project frames the question as a binary classification task:

- **Target**: viral = 1 if playCount > 100,000, else 0
- **Key constraint**: only features known at publication time are used — no likes, shares, or comments

---

## Dataset

Two TikTok datasets merged vertically:

- tiktok_dataset.csv — main trending video dataset
- meta_data.csv — supplementary metadata

**Pre-publication features retained:**

| Category | Features |
|---|---|
| Content | text (caption), hashtags, mentions |
| Timing | createTime |
| Author | authorMeta.verified, authorMeta.name |
| Music | musicMeta.musicId, musicMeta.musicOriginal, musicMeta.musicAuthor |
| Video | videoMeta.duration |

---

## ML Pipeline

```
Raw Data (tiktok_dataset.csv + meta_data.csv)
         |
         v
   Data Merging & Cleaning
   (drop nulls, strip whitespace, remove empty captions)
         |
         v
   Target Creation
   (viral = 1 if playCount > 100,000)
         |
         v
   Feature Engineering
   (hashtag count, description length, hour/day posted,
    author verified, music originality, top-author encoding)
         |
         v
   Statistical Analysis
   (Correlation matrix, VIF, PCA)
         |
         v
   StandardScaler normalization
         |
         v
   Logistic Regression (L1 & L2)
   + 5-fold Cross-Validation (ROC AUC)
```

---

## Feature Engineering

| Feature | Description |
|---|---|
| description_length | Number of characters in the video caption |
| n_hashtags | Count of hashtags used |
| has_hashtags | Binary indicator (1 if hashtags present) |
| author_verified | Whether the account is verified (0/1) |
| music_is_original | Whether the audio is original (0/1) |
| hour_posted | Hour of publication (0-23) |
| day_of_week | Day of week (0=Monday, 6=Sunday) |
| musicAuthor_* | One-hot encoding of the top 10 music authors |

---

## Statistical Analysis

Before modeling, rigorous multicollinearity and feature structure analysis is performed:

- **Correlation matrix** — identifies redundant feature pairs (threshold: |rho| > 0.7)
- **VIF (Variance Inflation Factor)** — detects and flags multicollinearity
- **PCA** — analyzes explained variance to understand dimensionality of feature space

---

## Models & Evaluation

Two penalized logistic regression models compared via cross-validation:

| Model | Regularization | Solver | Evaluation |
|---|---|---|---|
| Ridge | L2 | lbfgs | ROC AUC (5-fold CV) |
| Lasso | L1 | saga | ROC AUC (5-fold CV) |

Feature importance is assessed by ranking the top-10 absolute coefficients for each model.

---

## Project Structure

```
ML_PROJECT/
├── Tiktok_Viral.py                          # Full pipeline: cleaning -> features -> models
├── test.py                                  # Experimental tests and sanity checks
├── tiktok_dataset.csv                       # Raw TikTok dataset
├── meta_data.csv                            # Supplementary metadata
├── trending_converted.csv                   # Preprocessed trending data
├── Trending_pre_publication_with_target.csv # Filtered dataset with viral target
└── Trending_model_ready.csv                 # Final feature-engineered dataset
```

---

## Getting Started

```bash
git clone https://github.com/01Giuliano01/ML_PROJECT.git
cd ML_PROJECT
pip install pandas numpy scikit-learn statsmodels
python Tiktok_Viral.py
```

> **Note:** Update the file paths at the top of Tiktok_Viral.py to match your local directory.

---

## Tech Stack

| Library | Usage |
|---|---|
| pandas | Data loading, merging, feature engineering |
| numpy | Numerical operations |
| scikit-learn | Preprocessing, PCA, Logistic Regression, cross-validation |
| statsmodels | Variance Inflation Factor (VIF) computation |
