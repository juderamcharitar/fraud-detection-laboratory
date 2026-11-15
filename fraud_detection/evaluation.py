"""Held-out threshold analysis and score-level ensemble utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, average_precision_score


def evaluate_threshold(y, scores, threshold, false_alarm_cost=1.0, missed_fraud_cost=20.0):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    if y.ndim!=1 or scores.shape!=y.shape or len(y)==0 or not np.isfinite(scores).all() or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Expected aligned binary labels and finite anomaly scores')
    if false_alarm_cost<0 or missed_fraud_cost<0:
        raise ValueError('Costs must be nonnegative')
    alerts=scores>=float(threshold)
    tp=int(((alerts==1)&(y==1)).sum()); fp=int(((alerts==1)&(y==0)).sum())
    fn=int(((alerts==0)&(y==1)).sum()); tn=int(((alerts==0)&(y==0)).sum())
    p,r,f,_=precision_recall_fscore_support(y,alerts,average='binary',zero_division=0)
    return {'precision':float(p),'recall':float(r),'f1':float(f),'false_positive_rate':fp/max(1,fp+tn),
            'alert_rate':float(alerts.mean()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'total_cost':float(fp*false_alarm_cost+fn*missed_fraud_cost)}


def threshold_curve(y,scores,false_alarm_cost=1.0,missed_fraud_cost=20.0):
    scores=np.asarray(scores,dtype=float)
    thresholds=np.unique(np.quantile(scores,np.linspace(0,1,101)))
    thresholds=np.r_[np.nextafter(scores.max(),np.inf),thresholds]
    return pd.DataFrame([{'threshold':float(t),**evaluate_threshold(y,scores,t,false_alarm_cost,missed_fraud_cost)} for t in thresholds])


def recommended_threshold(y,scores,false_alarm_cost=1.0,missed_fraud_cost=20.0):
    """Retrospective exploratory optimum. NEVER treat test-selected threshold as unbiased test performance."""
    curve=threshold_curve(y,scores,false_alarm_cost,missed_fraud_cost)
    return curve.sort_values(['total_cost','false_positive_rate','threshold'],ascending=[True,True,False]).iloc[0]


def rank_ensemble(scored):
    """Average percentile ranks, so detector score scales cannot dominate."""
    if len(scored)<2: raise ValueError('Select at least two detectors')
    keys=list(scored)
    base=scored[keys[0]]
    for key in keys[1:]:
        if not base.index.equals(scored[key].index): raise ValueError('Detector rows must align')
    ranks=[scored[k]['anomaly_score'].rank(pct=True).to_numpy() for k in keys]
    return np.mean(ranks,axis=0)


def score_feature_perturbations(model, preprocessor, train, row, features, score_function):
    """Perturb-one-feature score sensitivity; descriptive and model-dependent, not causal."""
    baseline=float(score_function(preprocessor.transform(row[features]))[0])
    output=[]
    for f in features:
        altered=row[features].copy()
        if pd.api.types.is_numeric_dtype(train[f]):
            replacement=pd.to_numeric(train[f],errors='coerce').median()
        else:
            modes=train[f].mode(dropna=True)
            replacement=modes.iloc[0] if len(modes) else 'missing'
        altered[f]=replacement
        changed=float(score_function(preprocessor.transform(altered))[0])
        output.append({'feature':f,'score_change_if_replaced':baseline-changed})
    return pd.DataFrame(output).sort_values('score_change_if_replaced',ascending=False)

def robustness_trials(df, features, label, seeds=(11,23,37), contamination=.04, algorithms=None):
    """Repeat held-out experiments across random splits; descriptive, not independent datasets."""
    from .advanced import run_comparison
    rows=[]
    for seed in seeds:
        metrics,_,_=run_comparison(df,features,label,contamination,int(seed),algorithms)
        for row in metrics.to_dict('records'):
            rows.append({'seed':int(seed),**row})
    return pd.DataFrame(rows)
