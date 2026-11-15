import numpy as np
import pandas as pd
import pytest
from fraud_detection.evaluation import evaluate_threshold,threshold_curve,recommended_threshold,rank_ensemble

def test_costs():
    r=evaluate_threshold([0,0,1,1],[.1,.8,.3,.9],.5,2,10)
    assert (r['tp'],r['fp'],r['fn'],r['tn'],r['total_cost'])==(1,1,1,1,12)

def test_curve():
    y=[0,0,1,1]; scores=[.1,.2,.8,.9]
    c=threshold_curve(y,scores)
    assert len(c)>10
    assert recommended_threshold(y,scores)['total_cost']==0

def test_bad_input():
    with pytest.raises(ValueError): evaluate_threshold([0,2],[.1,.2],.2)
    with pytest.raises(ValueError): evaluate_threshold([0,1],[.1,float('nan')],.2)
    with pytest.raises(ValueError): evaluate_threshold([0,1],[.1,.2],.2,-1,1)

def test_rank_ensemble():
    a=pd.DataFrame({'anomaly_score':[1,2,3]},index=[1,2,3]); b=pd.DataFrame({'anomaly_score':[10,20,30]},index=[1,2,3])
    assert np.allclose(rank_ensemble({'a':a,'b':b}),[1/3,2/3,1])
    with pytest.raises(ValueError): rank_ensemble({'a':a})
    with pytest.raises(ValueError): rank_ensemble({'a':a,'b':b.iloc[::-1]})

def test_robustness():
    from fraud_detection.core import synthetic_transactions
    from fraud_detection.evaluation import robustness_trials
    df=synthetic_transactions(n=200,seed=42)
    out=robustness_trials(df,['amount','hour','velocity_24h','distance_km','merchant_risk'],'is_fraud',seeds=(1,2),algorithms=['Isolation Forest'])
    assert len(out)==2 and set(out.seed)=={1,2}
