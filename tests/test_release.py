import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from fraud_detection.core import synthetic_transactions
from fraud_detection.release import run_release, predict_new, export_bundle, import_trusted_bundle, feature_sensitivity

@pytest.fixture(scope='module')
def fitted():
    frame=synthetic_transactions(350, seed=17)
    report, scored, artifact, train=run_release(frame, ['amount','hour','velocity_24h','distance_km','merchant_risk'], seed=17)
    return frame, report, scored, artifact, train

def test_partition_counts(fitted):
    frame, report, scored, artifact, train=fitted
    assert sum(report['split_sizes'].values())==len(frame)
    assert report['split_sizes']=={'train':210,'calibration':70,'test':70}
    assert len(scored)==70

def test_report_and_metrics(fitted):
    _, report, scored, _, _=fitted
    assert report['threshold_method'].startswith('Cost minimum')
    assert 'precision' in report['test']
    assert len(report['dataset_fingerprint'])==64
    assert np.array_equal(scored.alert.to_numpy(),(scored.anomaly_score>=report['threshold']).astype(int))

def test_roundtrip(fitted):
    pytest.importorskip('skops.io')
    _, report, scored, artifact, _=fitted
    raw=export_bundle(artifact,report)
    loaded=import_trusted_bundle(raw)
    before=predict_new(artifact,scored)
    after=predict_new(loaded,scored)
    assert np.allclose(before.anomaly_score,after.anomaly_score)
    assert before.alert.equals(after.alert)

def test_missing_features(fitted):
    _, _, scored, artifact, _=fitted
    with pytest.raises(ValueError,match='Missing required'):
        predict_new(artifact,scored.drop(columns=['amount']))

def test_corrupted_bundle(fitted):
    pytest.importorskip('skops.io')
    _, report, _, artifact, _=fitted
    raw=export_bundle(artifact,report)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files={n:z.read(n) for n in z.namelist()}
    files['manifest.json']=files['manifest.json'].replace(b'fraud-lab-skops-v1', b'fraud-lab-skops-v0')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,data in files.items():z.writestr(name,data)
    with pytest.raises(ValueError,match='format'):
        import_trusted_bundle(stream.getvalue())

def test_duplicate_target_rejected():
    df=synthetic_transactions(200)
    df['fraud_copy']=df.is_fraud
    with pytest.raises(ValueError,match='duplicates'):
        run_release(df,['amount','fraud_copy'],'is_fraud')

def test_chronological(fitted):
    frame,_,_,_,_=fitted
    report, scored, _,_=run_release(frame,['amount','hour','velocity_24h','distance_km','merchant_risk'],chronological=True)
    assert scored.index.min()==280
    assert report['chronological_split']

def test_unlabeled(fitted):
    frame,_,_,_,_=fitted
    report, scored, _,_=run_release(frame.drop(columns=['is_fraud']),['amount','hour','velocity_24h','distance_km','merchant_risk'],label=None)
    assert report['threshold_method'].startswith('Train-only')
    assert 'precision' not in report['test']

def test_sensitivity(fitted):
    _,_,scored,artifact,train=fitted
    result=feature_sensitivity(artifact,train,scored.iloc[[0]])
    assert len(result)==5
    assert result.score_change_when_replaced.notna().all()


def test_legacy_pickle_bundle_rejected():
    from fraud_detection.release import _checked_zip
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('trusted_model.joblib',b'not a real model')
        z.writestr('report.json','{}')
        z.writestr('sha256.txt','bad')
    with pytest.raises(ValueError,match='legacy pickle'):
        _checked_zip(stream.getvalue())

def test_oversized_bundle_rejected():
    from fraud_detection.release import _checked_zip
    with pytest.raises(ValueError,match='too large'):
        _checked_zip(b'x'*30_000_001)
