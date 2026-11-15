"""Three-way, leakage-conscious anomaly research and trusted local model bundles."""
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from .advanced import validate_dataset
from .evaluation import recommended_threshold, evaluate_threshold

ALGORITHMS = ('Isolation Forest', 'Local Outlier Factor', 'Autoencoder')

def _preprocessor(frame, features):
    num = [f for f in features if pd.api.types.is_numeric_dtype(frame[f])]
    cat = [f for f in features if f not in num]
    transformers = []
    if num:
        transformers.append(('numeric', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num))
    if cat:
        transformers.append(('category', Pipeline([('fill', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False))]), cat))
    return ColumnTransformer(transformers, sparse_threshold=0)

def _detector(algorithm, seed, width):
    if algorithm == 'Isolation Forest':
        return IsolationForest(n_estimators=150, random_state=seed, n_jobs=-1)
    if algorithm == 'Local Outlier Factor':
        return LocalOutlierFactor(n_neighbors=20, novelty=True)
    if algorithm == 'Autoencoder':
        w = max(4, min(32, width * 2))
        return MLPRegressor(hidden_layer_sizes=(w, max(2, w // 2), w), random_state=seed, max_iter=180, early_stopping=True)
    raise ValueError('Unknown detector')

def anomaly_score(model, x, algorithm):
    if algorithm == 'Autoencoder':
        return np.mean((x - model.predict(x)) ** 2, axis=1)
    return -model.score_samples(x)

def _metrics(y, scores, threshold, fp_cost, fn_cost):
    alerts = (scores >= threshold).astype(int)
    if y is None:
        return {'alert_rate': float(alerts.mean())}
    metrics = evaluate_threshold(y, scores, threshold, fp_cost, fn_cost)
    if len(np.unique(y)) == 2:
        metrics['average_precision'] = float(average_precision_score(y, scores))
        metrics['roc_auc'] = float(roc_auc_score(y, scores))
    return metrics

def _split(frame, label, seed, chronological=False):
    if chronological:
        a = int(len(frame) * .6)
        b = int(len(frame) * .8)
        return frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]
    indices = np.arange(len(frame))
    def stratify(ix):
        if not label:
            return None
        counts = frame.iloc[ix][label].value_counts()
        return frame.iloc[ix][label] if len(counts) == 2 and counts.min() >= 3 else None
    train_ix, rest_ix = train_test_split(indices, test_size=.4, random_state=seed, stratify=stratify(indices))
    cal_ix, test_ix = train_test_split(rest_ix, test_size=.5, random_state=seed + 1, stratify=stratify(rest_ix))
    return frame.iloc[train_ix], frame.iloc[cal_ix], frame.iloc[test_ix]

def run_release(df, features, label='is_fraud', algorithm='Isolation Forest', seed=42, contamination=.04, false_alarm_cost=1., missed_fraud_cost=20., chronological=False):
    """60/20/20 split; fit preprocessing/model on train, select threshold on calibration, test once."""
    validate_dataset(df, features, label)
    if len(df) < 150:
        raise ValueError('v1.0 requires at least 150 rows for three-way splitting')
    if not 0 < contamination < .5 or false_alarm_cost < 0 or missed_fraud_cost < 0:
        raise ValueError('Invalid anomaly fraction or costs')
    if algorithm not in ALGORITHMS:
        raise ValueError('Unknown detector')
    # Explicit target-copy detection, including inversions; does not catch semantic or temporal leakage.
    if label:
        for feature in features:
            if df[feature].equals(df[label]):
                raise ValueError(f'{feature} duplicates the fraud label')
            if pd.api.types.is_numeric_dtype(df[feature]):
                vals = pd.to_numeric(df[feature], errors='coerce')
                if vals.notna().all() and ((vals == 1 - df[label]).all()):
                    raise ValueError(f'{feature} inversely duplicates the fraud label')
    train, cal, test = _split(df, label, int(seed), chronological)
    if len(train) < 50 or len(cal) < 20 or len(test) < 20:
        raise ValueError('Not enough rows in one of the data partitions')
    pre = _preprocessor(train, features)
    xt = np.asarray(pre.fit_transform(train[features]), dtype=float)
    xc = np.asarray(pre.transform(cal[features]), dtype=float)
    xe = np.asarray(pre.transform(test[features]), dtype=float)
    if xt.shape[1] > 250 or not np.isfinite(xt).all() or not np.isfinite(xc).all() or not np.isfinite(xe).all():
        raise ValueError('Encoded data invalid or exceeds 250 columns')
    model = _detector(algorithm, int(seed), xt.shape[1])
    if algorithm == 'Autoencoder':
        model.fit(xt, xt)
    else:
        model.fit(xt)
    train_scores = anomaly_score(model, xt, algorithm)
    cal_scores = anomaly_score(model, xc, algorithm)
    test_scores = anomaly_score(model, xe, algorithm)
    cal_y = cal[label].to_numpy(dtype=int) if label else None
    test_y = test[label].to_numpy(dtype=int) if label else None
    if cal_y is not None and len(np.unique(cal_y)) == 2:
        threshold = float(recommended_threshold(cal_y, cal_scores, false_alarm_cost, missed_fraud_cost)['threshold'])
        method = 'Cost minimum on calibration labels'
    else:
        threshold = float(np.quantile(train_scores, 1 - contamination))
        method = 'Train-only anomaly quantile (calibration lacks both classes)'
    output = test.copy()
    output['anomaly_score'] = test_scores
    output['alert'] = (test_scores >= threshold).astype(int)
    report = {
        'version': '1.0.0', 'algorithm': algorithm, 'features': list(features), 'label': label,
        'seed': int(seed), 'chronological_split': bool(chronological),
        'split_sizes': {'train': len(train), 'calibration': len(cal), 'test': len(test)},
        'threshold': threshold, 'threshold_method': method,
        'costs': {'false_alarm': float(false_alarm_cost), 'missed_fraud': float(missed_fraud_cost)},
        'calibration': _metrics(cal_y, cal_scores, threshold, false_alarm_cost, missed_fraud_cost),
        'test': _metrics(test_y, test_scores, threshold, false_alarm_cost, missed_fraud_cost),
        'dataset_fingerprint': hashlib.sha256(pd.util.hash_pandas_object(df[features], index=True).values.tobytes()).hexdigest(),
        'warnings': ['Synthetic data does not validate real-world performance.', 'Time or entity leakage requires domain-aware splitting.', 'Threshold selected on calibration only; test labels are not used for selection.']
    }
    artifact = {'preprocessor': pre, 'detector': model, 'algorithm': algorithm, 'features': list(features), 'threshold': threshold}
    return report, output, artifact, train

def predict_new(artifact, df):
    missing = [f for f in artifact['features'] if f not in df]
    if missing:
        raise ValueError('Missing required features: ' + ', '.join(missing))
    x = np.asarray(artifact['preprocessor'].transform(df[artifact['features']]), dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed data')
    scores = anomaly_score(artifact['detector'], x, artifact['algorithm'])
    return pd.DataFrame({'anomaly_score': scores, 'alert': (scores >= artifact['threshold']).astype(int)}, index=df.index)

# A deliberately narrow set of sklearn estimator/transformer classes used by this app.
# Never infer trusted types from an uploaded file: that defeats the allowlist.
SAFE_SKOPS_TYPES = frozenset({
    'sklearn.compose._column_transformer.ColumnTransformer',
    'sklearn.pipeline.Pipeline',
    'sklearn.impute._base.SimpleImputer',
    'sklearn.preprocessing._data.StandardScaler',
    'sklearn.preprocessing._encoders.OneHotEncoder',
    'sklearn.ensemble._iforest.IsolationForest',
    'sklearn.tree._classes.ExtraTreeRegressor',
    'sklearn.tree._tree.Tree',
    'sklearn.neighbors._lof.LocalOutlierFactor',
    'sklearn.neighbors._kd_tree.KDTree',
    'sklearn.neighbors._ball_tree.BallTree',
    'sklearn.neural_network._multilayer_perceptron.MLPRegressor',
    'numpy.dtype', 'numpy.random._pickle.__randomstate_ctor',
    'numpy.random._pickle.__generator_ctor',
})
MAX_BUNDLE_BYTES = 30_000_000
MAX_MODEL_BYTES = 25_000_000

def _checked_zip(data):
    if not isinstance(data, bytes) or len(data) > MAX_BUNDLE_BYTES:
        raise ValueError('Invalid bundle or bundle too large')
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if set(z.namelist()) != {'manifest.json', 'model.skops'} or len(z.namelist()) != 2:
                raise ValueError('Unsupported bundle format (legacy pickle bundles are rejected)')
            for info in z.infolist():
                if info.file_size > MAX_MODEL_BYTES or info.compress_size > MAX_BUNDLE_BYTES:
                    raise ValueError('Bundle entry too large')
            manifest_bytes = z.read('manifest.json')
            model_bytes = z.read('model.skops')
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError('Invalid model bundle ZIP') from exc
    if len(manifest_bytes) > 100_000 or len(model_bytes) > MAX_MODEL_BYTES:
        raise ValueError('Bundle payload too large')
    return manifest_bytes, model_bytes

def export_bundle(artifact, report):
    """Serialize with skops; no pickle/joblib deserialization is used."""
    import skops.io as sio
    model_bytes = sio.dumps(artifact)
    if len(model_bytes) > MAX_MODEL_BYTES:
        raise ValueError('Model too large to export')
    manifest = {
        'format': 'fraud-lab-skops-v1', 'version': '1.1.0',
        'model_sha256': hashlib.sha256(model_bytes).hexdigest(),
        'report': report,
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, default=float))
        z.writestr('model.skops', model_bytes)
    if out.tell() > MAX_BUNDLE_BYTES:
        raise ValueError('Bundle too large')
    return out.getvalue()

def import_safe_bundle(data):
    """Load only explicitly allowed sklearn object types; reject legacy pickle."""
    import skops.io as sio
    manifest_bytes, model_bytes = _checked_zip(data)
    try:
        manifest = json.loads(manifest_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('Invalid manifest') from exc
    if not isinstance(manifest, dict) or manifest.get('format') != 'fraud-lab-skops-v1':
        raise ValueError('Unsupported model format')
    if hashlib.sha256(model_bytes).hexdigest() != manifest.get('model_sha256'):
        raise ValueError('Bundle checksum mismatch')
    untrusted = set(sio.get_untrusted_types(data=model_bytes))
    unexpected = untrusted - SAFE_SKOPS_TYPES
    if unexpected:
        raise ValueError('Unsupported model object types: ' + ', '.join(sorted(unexpected)[:5]))
    artifact = sio.loads(model_bytes, trusted=list(untrusted))
    if not isinstance(artifact, dict) or set(artifact) != {'preprocessor','detector','features','algorithm','threshold'}:
        raise ValueError('Invalid model artifact')
    if artifact['algorithm'] not in ALGORITHMS or not isinstance(artifact['features'], list) or not artifact['features']:
        raise ValueError('Invalid model metadata')
    if any(not isinstance(f, str) or not f or len(f)>200 for f in artifact['features']):
        raise ValueError('Invalid model feature names')
    if not np.isfinite(float(artifact['threshold'])):
        raise ValueError('Invalid model threshold')
    if not isinstance(artifact['preprocessor'], ColumnTransformer):
        raise ValueError('Invalid preprocessor')
    expected_type = {'Isolation Forest': IsolationForest, 'Local Outlier Factor': LocalOutlierFactor, 'Autoencoder': MLPRegressor}[artifact['algorithm']]
    if not isinstance(artifact['detector'], expected_type):
        raise ValueError('Invalid detector')
    return artifact

# Backwards-compatible Python API name, but the legacy pickle format is rejected.
import_trusted_bundle = import_safe_bundle

def feature_sensitivity(artifact, training, row):
    """One-feature-at-a-time reference replacement; NOT causal attribution."""
    base = float(predict_new(artifact, row)['anomaly_score'].iloc[0])
    rows = []
    for f in artifact['features']:
        candidate = row.copy()
        if pd.api.types.is_numeric_dtype(training[f]):
            value = training[f].median()
        else:
            mode = training[f].mode(dropna=True)
            value = mode.iloc[0] if len(mode) else None
        candidate[f] = value
        altered = float(predict_new(artifact, candidate)['anomaly_score'].iloc[0])
        rows.append({'feature': f, 'score_change_when_replaced': base - altered})
    return pd.DataFrame(rows).sort_values('score_change_when_replaced', ascending=False)

