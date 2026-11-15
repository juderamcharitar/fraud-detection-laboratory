import numpy as np
import pytest
from fraud_detection.core import synthetic_transactions, fit_and_score, FEATURES


def test_reproducible():
    assert synthetic_transactions(seed=7).equals(synthetic_transactions(seed=7))


def test_schema():
    assert set(FEATURES + ['is_fraud']).issubset(synthetic_transactions().columns)


def test_run():
    scored, metrics = fit_and_score(synthetic_transactions(n=500))
    assert len(scored) == 125
    assert 0 <= metrics['precision'] <= 1
    assert 'anomaly_score' in scored


def test_invalid_rate():
    with pytest.raises(ValueError):
        synthetic_transactions(fraud_rate=0)


def test_invalid_contamination():
    with pytest.raises(ValueError):
        fit_and_score(synthetic_transactions(), contamination=0)


def test_missing_features():
    with pytest.raises(ValueError):
        fit_and_score(synthetic_transactions().drop(columns=['amount']))


def test_missing_values():
    df = synthetic_transactions()
    df.loc[0, 'amount'] = np.nan
    with pytest.raises(ValueError):
        fit_and_score(df)
