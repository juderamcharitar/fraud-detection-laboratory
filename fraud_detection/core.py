"""Synthetic transaction data and unsupervised anomaly detection."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['amount', 'hour', 'velocity_24h', 'distance_km', 'merchant_risk']


def synthetic_transactions(n=3000, fraud_rate=0.04, seed=42):
    if n < 100 or not 0 < fraud_rate < 0.5:
        raise ValueError('n must be >=100 and fraud_rate must be between 0 and 0.5')
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    amount = rng.lognormal(3.5, 0.9, n)
    hour = rng.integers(0, 24, n)
    velocity = rng.poisson(2, n)
    distance = rng.exponential(12, n)
    risk = rng.beta(2, 8, n)
    k = int(y.sum())
    amount[y == 1] *= rng.uniform(2, 8, k)
    velocity[y == 1] += rng.poisson(8, k)
    distance[y == 1] += rng.exponential(100, k)
    risk[y == 1] = rng.beta(7, 2, k)
    df = pd.DataFrame({'amount': amount, 'hour': hour, 'velocity_24h': velocity,
                       'distance_km': distance, 'merchant_risk': risk, 'is_fraud': y})
    return df


def evaluate(y_true, scores, alerts):
    p, r, f, _ = precision_recall_fscore_support(y_true, alerts, average='binary', zero_division=0)
    result = {'precision': float(p), 'recall': float(r), 'f1': float(f),
              'alert_rate': float(np.mean(alerts)), 'base_rate': float(np.mean(y_true))}
    if len(np.unique(y_true)) == 2:
        result['average_precision'] = float(average_precision_score(y_true, scores))
        result['roc_auc'] = float(roc_auc_score(y_true, scores))
    return result


def fit_and_score(df, contamination=0.04, seed=42):
    if not 0 < contamination < 0.5:
        raise ValueError('contamination must be between 0 and 0.5')
    if not set(FEATURES).issubset(df.columns):
        raise ValueError('Required feature columns are missing')
    if df[FEATURES].isna().any().any() or not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Features must be finite and nonmissing')
    if len(df) < 30:
        raise ValueError('At least 30 transactions are required')
    # Labels are never supplied to the model; use a held-out set for honest evaluation.
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.25,
                                           random_state=seed, stratify=df['is_fraud'] if 'is_fraud' in df and df['is_fraud'].value_counts().min() >= 2 else None)
    model = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, contamination=contamination, random_state=seed))
    model.fit(df.iloc[idx_train][FEATURES])
    test = df.iloc[idx_test].copy()
    scores = -model.decision_function(test[FEATURES])
    alerts = (model.predict(test[FEATURES]) == -1).astype(int)
    test['anomaly_score'] = scores
    test['alert'] = alerts
    metrics = evaluate(test['is_fraud'].to_numpy(), scores, alerts) if 'is_fraud' in test else None
    return test, metrics
