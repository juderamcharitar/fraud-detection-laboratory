import numpy as np
import pandas as pd
import pytest
from fraud_detection.core import synthetic_transactions
from fraud_detection.advanced import run_comparison, validate_dataset, feature_deviations

@pytest.fixture(scope="module")
def dataset():
    return synthetic_transactions(n=240,seed=11)

def test_all_algorithms(dataset):
    results, scored, _ = run_comparison(dataset,["amount","hour","velocity_24h","distance_km","merchant_risk"],"is_fraud",seed=4)
    assert len(results)==3
    assert all(len(x)==60 for x in scored.values())
    assert all("alert" in x for x in scored.values())

def test_no_label(dataset):
    results,_,_=run_comparison(dataset,["amount","hour"],None,algorithms=["Isolation Forest"])
    assert "precision" not in results.columns

def test_categorical_missing(dataset):
    data=dataset.copy()
    data["merchant"]=np.where(np.arange(len(data))%2,"A","B")
    data.loc[0,"merchant"]=None
    result,_,_=run_comparison(data,["merchant","amount"],"is_fraud",algorithms=["Isolation Forest"])
    assert len(result)==1

def test_label_not_feature(dataset):
    with pytest.raises(ValueError):
        validate_dataset(dataset,["amount","is_fraud"],"is_fraud")

def test_identifier_rejected(dataset):
    data=dataset.copy()
    data["transaction_id"]=np.arange(len(data))
    with pytest.raises(ValueError):
        validate_dataset(data,["transaction_id"],"is_fraud")

def test_bad_label(dataset):
    data=dataset.copy()
    data["is_fraud"]=2
    with pytest.raises(ValueError):
        validate_dataset(data,["amount"],"is_fraud")

def test_deviations(dataset):
    output=feature_deviations(dataset,["amount","hour"],dataset.index[0])
    assert set(output["feature"])=={"amount","hour"}
