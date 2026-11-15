"""Comparable anomaly detectors with train-only preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score

def validate_dataset(df, features, label=None):
    if not isinstance(df, pd.DataFrame) or len(df) < 80:
        raise ValueError("At least 80 rows are required")
    if len(df) > 100000:
        raise ValueError("Maximum 100,000 rows")
    if not features or len(features) > 30 or len(set(features)) != len(features):
        raise ValueError("Select 1–30 unique features")
    if any(f not in df.columns for f in features):
        raise ValueError("Unknown feature")
    if label and (label not in df or label in features):
        raise ValueError("Label must exist and must not be an input feature")
    if label:
        values = df[label].dropna().unique()
        if len(values) != 2 or not set(values).issubset({0,1,False,True}):
            raise ValueError("Optional fraud label must contain 0 and 1")
        if df[label].isna().any():
            raise ValueError("Fraud labels cannot be missing")
    for f in features:
        if f.lower() in {"id", "transaction_id", "customer_id", "account_id", "user_id"} or f.lower().endswith("_id"):
            raise ValueError(f"Feature {f} looks like an identifier")
        if df[f].isna().all():
            raise ValueError(f"Feature {f} is entirely missing")
        if not pd.api.types.is_numeric_dtype(df[f]) and df[f].nunique(dropna=True) > max(100, int(len(df)*0.9)):
            raise ValueError(f"Feature {f} is nearly unique; remove identifiers or high-cardinality fields")
    return True

def run_comparison(df, features, label=None, contamination=0.04, seed=42, algorithms=None):
    validate_dataset(df, features, label)
    if not 0 < contamination < 0.5:
        raise ValueError("Expected anomaly fraction must be between 0 and 0.5")
    algorithms = algorithms or ["Isolation Forest", "Local Outlier Factor", "Autoencoder"]
    if not algorithms or any(a not in ("Isolation Forest","Local Outlier Factor","Autoencoder") for a in algorithms):
        raise ValueError("Unknown algorithm")
    idx_train, idx_test = train_test_split(np.arange(len(df)),test_size=0.25,random_state=seed)
    train, test = df.iloc[idx_train],df.iloc[idx_test].copy()
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    categorical = [f for f in features if f not in numeric]
    # Fit imputation, encoding, scaling on training data ONLY.
    transforms = []
    if numeric:
        transforms.append(("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric))
    if categorical:
        transforms.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",max_categories=20,sparse_output=False))]),categorical))
    pre = ColumnTransformer(transforms,sparse_threshold=0)
    xtrain = np.asarray(pre.fit_transform(train[features]),dtype=float)
    xtest = np.asarray(pre.transform(test[features]),dtype=float)
    if xtrain.shape[1] > 250:
        raise ValueError("Too many encoded features (maximum 250)")
    results, scored = [], {}
    for algorithm in algorithms:
        if algorithm == "Isolation Forest":
            model = IsolationForest(n_estimators=150,random_state=seed,n_jobs=-1)
            model.fit(xtrain)
            raw_train = -model.score_samples(xtrain)
            scores = -model.score_samples(xtest)
        elif algorithm == "Local Outlier Factor":
            model = LocalOutlierFactor(n_neighbors=min(20,len(xtrain)-1),novelty=True)
            model.fit(xtrain)
            raw_train = -model.score_samples(xtrain)
            scores = -model.score_samples(xtest)
        else:
            # MLP reconstruction network: autoencoder-style anomaly model.
            # Uses scikit-learn's MLPRegressor; not a deep-learning framework.
            width = max(4,min(32,xtrain.shape[1]*2))
            model = MLPRegressor(hidden_layer_sizes=(width,max(2,width//2),width),
                                 random_state=seed,max_iter=180,early_stopping=True,
                                 validation_fraction=0.15)
            model.fit(xtrain,xtrain)
            raw_train = np.mean((xtrain-model.predict(xtrain))**2,axis=1)
            scores = np.mean((xtest-model.predict(xtest))**2,axis=1)
        threshold = float(np.quantile(raw_train,1-contamination))
        alerts = (scores >= threshold).astype(int)
        output = test.copy()
        output["anomaly_score"] = scores
        output["alert"] = alerts
        scored[algorithm] = output
        row = {"algorithm":algorithm,"alert_rate":float(alerts.mean()),"threshold":threshold}
        if label:
            y = test[label].astype(int).to_numpy()
            precision,recall,f1,_ = precision_recall_fscore_support(y,alerts,average="binary",zero_division=0)
            row.update(precision=float(precision),recall=float(recall),f1=float(f1),
                       false_positive_rate=float(((alerts==1)&(y==0)).sum()/max(1,(y==0).sum())))
            if len(np.unique(y))==2:
                row["average_precision"] = float(average_precision_score(y,scores))
                row["roc_auc"] = float(roc_auc_score(y,scores))
        results.append(row)
    return pd.DataFrame(results),scored,pre

def feature_deviations(df, features, row_index):
    """Descriptive deviations only; not causal explanations of model scores."""
    row = df.loc[row_index]
    output=[]
    for f in features:
        if pd.api.types.is_numeric_dtype(df[f]):
            col=pd.to_numeric(df[f],errors="coerce")
            sd=col.std()
            if pd.notna(sd) and sd>0 and pd.notna(row[f]):
                output.append({"feature":f,"deviation":float(abs((row[f]-col.median())/sd))})
    return pd.DataFrame(output).sort_values("deviation",ascending=False) if output else pd.DataFrame(columns=["feature","deviation"])
