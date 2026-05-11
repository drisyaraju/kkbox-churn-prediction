# KKBox Music Streaming — Churn Prediction

A complete end-to-end machine learning pipeline to predict customer churn for KKBox, a music streaming service based in Taiwan. Built on the real-world dataset from the **WSDM Cup 2018 Kaggle Competition**.

**Final ROC-AUC: 0.9789** on hold-out test set.

---

## What this project is

KKBox wants to predict whether a subscriber will **not renew their subscription within 30 days** of expiry. This is a binary classification problem on large, messy, relational data — four tables that need to be joined, cleaned, and engineered before any modeling can happen.

- `is_churn = 1` → user did not renew (churned)
- `is_churn = 0` → user renewed

Churn rate is ~6–7%, making this a severely imbalanced classification problem. Accuracy is a useless metric here — the project uses **ROC-AUC** throughout.

---

## Dataset

Downloaded from: [kaggle.com/competitions/kkbox-churn-prediction-challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)

| File | Size | Description |
|---|---|---|
| `train_v2.csv` | Small | Labels — `msno` (user ID) + `is_churn` |
| `members_v3.csv` | ~30 MB | Demographics — age, city, gender, registration info |
| `transactions_v2.csv` | ~200 MB | Subscription events per user |
| `user_logs_v2.csv` | ~3.5 GB | Daily listening behaviour — processed in chunks |

> The dataset files are not included in this repository due to size. Download them from Kaggle and place them in the project root.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.9789** |
| Recall (churned) | 0.88 |
| Precision (churned) | 0.70 |
| Best iteration (early stopping) | 40 |

### Note on competition leaderboard scores

The Kaggle leaderboard top scores were ~0.90–0.97. The difference is expected:

- **This project** evaluates on a random 20% hold-out from the same dataset — train and test users share similar distributions and the same time period.
- **The competition** evaluated on a hidden future test set with users whose churn decisions happened after the training window — a harder, more realistic evaluation.

This gap reflects **distribution shift** between training time and future users, not model overfitting. A strict time-based split would require the labeled competition test set which is not publicly available.

---

## Project structure

```
kkbox-churn/
├── 01_explore.ipynb        # Data loading, shape checks, null analysis, churn rate
├── 02_transactions.ipynb   # Transaction aggregation → txn_agg.csv
├── 03_user_logs.ipynb      # Chunk processing of 3.5GB logs → log_agg.csv
├── 04_merge.ipynb          # Relational join of all 4 tables → merged_features.csv
├── 05_eda.ipynb            # 12 visualizations — distributions, correlations, pair plots
├── 06_model.ipynb          # LightGBM training, evaluation, feature importance
├── requirements.txt
└── README.md
```

---

## 6-notebook pipeline

### 01 — Explore
Load `train_v2.csv` and `members_v3.csv`. Check shapes, nulls, dtypes. Confirm churn rate and identify data quality issues — particularly the `bd` (age) column which has severe outliers (negatives, values in thousands, 0 = missing).

### 02 — Transactions
Load `transactions_v2.csv`, sort by date, group by user. Aggregate to one row per user: last price paid, last auto-renew status, total cancellations, number of transactions, last expiry date.

### 03 — User Logs
The 3.5GB `user_logs_v2.csv` file cannot be loaded into memory directly. Processed in 500k-row chunks, aggregated per user: total listening seconds, unique songs, active days, completion rate. Key note: the column name is `num_unq` — not `num_uniq`.

### 04 — Merge
Join all four tables on `msno` (user ID) using left joins anchored to the training labels. Clean age outliers, extract registration year, one-hot encode categoricals, fill nulls with 0.

### 05 — EDA
12 visualizations including:
- Class imbalance (bar + pie)
- Correlation heatmap of top 15 features
- Boxplots by churn status
- Pair plot (5 key features, sampled)
- Auto-renew vs churn rate — the strongest single signal
- KDE distributions, age groups, cancellation history, registration year trends
- Engagement scatter with density contours

### 06 — Model
LightGBM binary classifier with:
- Stratified 80/20 train/test split
- `scale_pos_weight` to handle class imbalance (~14x)
- Early stopping (50 rounds) on AUC
- Evaluation: ROC-AUC, classification report, confusion matrix, ROC curve
- Top 20 feature importances

---

## Key findings

| Feature | Insight |
|---|---|
| `registration_year` | Strongest overall signal — different user cohorts have distinct churn patterns |
| `last_price` | Higher plan price → higher churn — price sensitivity is real |
| `num_transactions` | More transactions = longer tenure = lower churn |
| `last_auto_renew` | Users who disable auto-renew are signalling intent to leave |
| `active_days` / `total_secs` | Disengaged users (low listening) churn far more |
| `total_cancels` | Even 1 historical cancellation is a strong churn warning |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/kkbox-churn.git
cd kkbox-churn
pip install -r requirements.txt
```

Download the dataset from Kaggle and place all CSV files in the project root. Then run notebooks 01 through 06 in order.

---

## Requirements

```
pandas
numpy
lightgbm
scikit-learn
matplotlib
seaborn
imbalanced-learn
jupyter
openpyxl
scipy
joblib
```

---

## Technical notes

- `user_logs_v2.csv` must always be loaded with `chunksize=500_000` — never load it directly into memory
- Age column `bd` filtered to values between 10 and 80 only; all other values treated as missing
- `last_expire` dropped before modeling — saved as a date string in CSV, LightGBM requires numeric types
- LightGBM can be switched to GPU with `device='gpu'` for faster training
- Model achieves best iteration at round 40, suggesting features are highly informative and clean

---

## Author

Built as a portfolio project to demonstrate end-to-end ML engineering on real-world relational data at scale.
