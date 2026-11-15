import io
import pandas as pd
import numpy as np
import streamlit as st
from fraud_detection.core import synthetic_transactions
from fraud_detection.advanced import run_comparison, feature_deviations
from fraud_detection.evaluation import evaluate_threshold, threshold_curve, recommended_threshold, rank_ensemble, robustness_trials

st.set_page_config(page_title="Fraud Detection Laboratory",layout="wide")
st.title("Fraud Detection Laboratory")
st.caption("Project #4 Â· v1.0 Â· Unsupervised anomaly research. An alert is not proof of fraud.")
# v1.0 research workflow; original v0.3 experiments remain available.
from fraud_detection.release import run_release, predict_new, export_bundle, import_trusted_bundle, feature_sensitivity
workflow=st.sidebar.radio("Workspace", ["v1.0 validated research", "v0.3 comparison laboratory", "Safe model inference"])
if workflow == "v1.0 validated research":
    st.subheader("Train / calibrate / test â€” independent partitions")
    source=st.radio("Dataset", ["Synthetic", "Upload CSV / Excel / Parquet"], horizontal=True)
    if source == "Synthetic":
        n=st.slider("Transactions", 500, 10000, 3000, 500)
        frame=synthetic_transactions(n=n)
        target="is_fraud"
        columns=["amount", "hour", "velocity_24h", "distance_km", "merchant_risk"]
    else:
        file=st.file_uploader("Upload transaction dataset", type=["csv", "xlsx", "parquet"])
        if file is None:
            st.info("Upload a dataset to begin.")
            st.stop()
        if file.size>20_000_000:
            st.error("Maximum upload size: 20 MB")
            st.stop()
        try:
            frame=pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file) if file.name.lower().endswith('.xlsx') else pd.read_parquet(file)
        except Exception as exc:
            st.error(f"Cannot parse file: {exc}")
            st.stop()
        target_choice=st.selectbox("Fraud label", ["None"]+list(frame.columns), index=(list(frame.columns).index("is_fraud")+1 if "is_fraud" in frame else 0))
        target=None if target_choice=="None" else target_choice
        columns=st.multiselect("Features (exclude IDs, target copies, future data)", [c for c in frame if c!=target], default=[c for c in frame if c!=target and not c.lower().endswith('id')][:5])
    st.dataframe(frame.head(15), width='stretch')
    a,b,c=st.columns(3)
    with a:
        algo=st.selectbox("Detector", ["Isolation Forest", "Local Outlier Factor", "Autoencoder"])
        seed_v1=st.number_input("Random seed", 0, 999999, 42)
    with b:
        frac=st.slider("Expected anomaly fraction", .01, .20, .04, .01)
        chronological=st.checkbox("Chronological split (already ordered rows)", value=False)
    with c:
        fp_cost=st.number_input("False alert cost", min_value=0., value=1.)
        fn_cost=st.number_input("Missed fraud cost", min_value=0., value=20.)
    st.caption("60% train, 20% calibration, 20% untouched test. Labels calibrate the threshold only on the calibration set. For chronological data, sort oldest to newest before upload. For repeated accounts, use an entity-aware split outside this demo.")
    if st.button("Train, calibrate, and evaluate", type="primary"):
        try:
            with st.spinner("Training and evaluating..."):
                report, scored_v1, artifact, train_v1=run_release(frame,columns,target,algo,int(seed_v1),frac,fp_cost,fn_cost,chronological)
            st.session_state['v1_results']=(report,scored_v1,artifact,train_v1)
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
    if 'v1_results' in st.session_state:
        report,scored_v1,artifact,train_v1=st.session_state['v1_results']
        st.subheader("Untouched test-set evaluation")
        st.json(report)
        st.dataframe(scored_v1.head(100), width='stretch')
        st.download_button("Download experiment report (JSON)", json.dumps(report,indent=2,default=float), file_name="fraud_experiment_report.json")
        st.download_button("Download test predictions", scored_v1.to_csv(index=False), file_name="fraud_test_predictions.csv")
        st.download_button("Download safe model bundle", export_bundle(artifact,report), file_name="fraud_model_safe.zip", mime="application/zip")
        st.subheader("Transaction-level sensitivity")
        st.caption("Reference-replacement score changes are model-dependent sensitivity checks, not causal explanations or fraud probabilities.")
        flagged=scored_v1[scored_v1.alert==1]
        if len(flagged):
            row_id=st.selectbox("Flagged transaction row",list(flagged.index[:100]))
            st.dataframe(feature_sensitivity(artifact,train_v1,scored_v1.loc[[row_id]]), width='stretch')
        else:
            st.info("No alerts at this threshold.")
    st.stop()
if workflow == "Safe model inference":
    st.warning("Model bundles use skops with an explicit allowed-type list; legacy pickle/joblib bundles are rejected. Only use models from sources you trust and verify.")
    bundle=st.file_uploader("Safe model ZIP", type=["zip"])
    new_data=st.file_uploader("New transactions CSV / Excel / Parquet", type=["csv", "xlsx", "parquet"])
    if st.button("Predict using saved model"):
        if not bundle or not new_data:
            st.error("Provide both files.")
        elif bundle.size>30_000_000 or new_data.size>20_000_000:
            st.error("File exceeds size limit.")
        else:
            try:
                model=import_trusted_bundle(bundle.getvalue())
                incoming=pd.read_csv(new_data) if new_data.name.lower().endswith('.csv') else pd.read_excel(new_data) if new_data.name.lower().endswith('.xlsx') else pd.read_parquet(new_data)
                predictions=predict_new(model,incoming)
                st.dataframe(predictions.head(100), width='stretch')
                st.download_button("Download predictions", predictions.to_csv(index=False),file_name="fraud_new_predictions.csv")
                st.caption("Anomaly scores are not calibrated fraud probabilities. Never use alerts as sole evidence of fraud.")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
    st.stop()

mode=st.sidebar.radio("Data source",["Synthetic transactions","Upload dataset"])
seed=st.sidebar.number_input("Random seed",min_value=0,max_value=999999,value=42)
contamination=st.sidebar.slider("Expected anomaly fraction",0.01,0.20,0.04,0.01)
if mode=="Synthetic transactions":
    n=st.sidebar.slider("Transactions",500,10000,3000,500)
    fraud_rate=st.sidebar.slider("Synthetic fraud prevalence",0.01,0.20,0.04,0.01)
    df=synthetic_transactions(n=n,fraud_rate=fraud_rate,seed=int(seed))
    label="is_fraud"
    features=["amount","hour","velocity_24h","distance_km","merchant_risk"]
else:
    upload=st.file_uploader("Upload CSV, Excel, or Parquet",type=["csv","xlsx","parquet"])
    if upload is None:
        st.info("Upload a dataset to begin. Fraud labels are optional.")
        st.stop()
    if upload.size>20_000_000:
        st.error("Maximum file size is 20 MB")
        st.stop()
    try:
        if upload.name.lower().endswith(".csv"):
            df=pd.read_csv(upload)
        elif upload.name.lower().endswith(".xlsx"):
            df=pd.read_excel(upload)
        else:
            df=pd.read_parquet(upload)
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        st.stop()
    label_option=st.selectbox("Fraud label (optional)",["None"]+list(df.columns),
                              index=(list(df.columns).index("is_fraud")+1 if "is_fraud" in df.columns else 0))
    label=None if label_option=="None" else label_option
    defaults=[c for c in df.columns if c!=label and df[c].nunique(dropna=True)<min(100,len(df)*0.9)][:5]
    features=st.multiselect("Input features (exclude IDs, outcomes and future information)",
                            [c for c in df.columns if c!=label],default=defaults)
st.subheader("Dataset preview")
st.dataframe(df.head(20),width='stretch')
st.caption(f"{len(df):,} rows Â· {len(features)} selected features Â· Label: {label or 'none'}")
algorithms=st.multiselect("Algorithms",["Isolation Forest","Local Outlier Factor","Autoencoder"],
                          default=["Isolation Forest","Local Outlier Factor","Autoencoder"])
if st.button("Run comparison",type="primary"):
    try:
        with st.spinner("Training detectors on the same split..."):
            comparison,scored,_=run_comparison(df,features,label,contamination,int(seed),algorithms)
        st.session_state["fraud_v02_result"]=(comparison,scored,features)
    except Exception as exc:
        st.error(f"Experiment could not run: {exc}")
        st.stop()
if "fraud_v02_result" in st.session_state:
    comparison,scored,used_features=st.session_state["fraud_v02_result"]
    st.subheader("Algorithm comparison")
    st.dataframe(comparison,width='stretch')
    st.caption("All detectors train without labels. Labels, when provided, are used only for held-out evaluation. A synthetic-data result does not establish real-world fraud performance.")
    st.subheader("Evaluation & explainability laboratory")
    with st.expander("Robustness: repeat across random splits"):
        st.caption("Runs three new train/test splits. Test samples overlap, so these are not independent external validations. Autoencoder runs can take longer.")
        if st.button("Run three-seed robustness check"):
            with st.spinner("Evaluating repeated splits..."):
                trials=robustness_trials(df,used_features,label,seeds=(11,23,37),contamination=contamination,algorithms=algorithms)
            st.dataframe(trials,width='stretch')
            numeric_cols=[c for c in ["precision","recall","f1","average_precision","roc_auc","alert_rate"] if c in trials]
            st.dataframe(trials.groupby("algorithm")[numeric_cols].agg(["mean","std"]))
            st.download_button("Download robustness results",trials.to_csv(index=False),file_name="robustness_trials.csv")
    st.caption("Threshold analysis below is retrospective on the held-out sample. Selecting a threshold using test labels creates optimistic bias: validate the chosen threshold on a fresh untouched dataset before deployment.")
    if len(scored)>1:
        if st.checkbox("Include percentile-rank ensemble",value=False):
            ensemble=scored[next(iter(scored))].copy()
            ensemble["anomaly_score"]=rank_ensemble(scored)
            ensemble["alert"]=(ensemble["anomaly_score"]>=ensemble["anomaly_score"].quantile(1-contamination)).astype(int)
            scored={**scored,"Rank ensemble":ensemble}
    chosen=st.selectbox("Inspect detector",list(scored))
    result=scored[chosen]
    if label is not None and label in result.columns:
        y=result[label].astype(int).to_numpy()
        scores=result["anomaly_score"].to_numpy()
        left,right=st.columns(2)
        with left:
            fp_cost=st.number_input("Cost of false alarm",min_value=0.0,value=1.0,step=1.0)
        with right:
            fn_cost=st.number_input("Cost of missed fraud",min_value=0.0,value=20.0,step=1.0)
        original=float(comparison.loc[comparison.algorithm==chosen,"threshold"].iloc[0]) if chosen in comparison.algorithm.values else float(np.quantile(scores,1-contamination))
        low,high=float(scores.min()),float(scores.max())
        if high>low:
            chosen_threshold=st.slider("Explore alert threshold (held-out data)",low,high,min(max(original,low),high))
        else:
            chosen_threshold=low
        current=evaluate_threshold(y,scores,chosen_threshold,fp_cost,fn_cost)
        st.write("Exploratory threshold metrics",current)
        curve=threshold_curve(y,scores,fp_cost,fn_cost)
        st.line_chart(curve.sort_values("threshold").set_index("threshold")[["precision","recall","f1"]])
        st.line_chart(curve.sort_values("threshold").set_index("threshold")[["total_cost"]])
        best=recommended_threshold(y,scores,fp_cost,fn_cost)
        st.caption(f"Exploratory minimum-cost threshold on these test labels: {best['threshold']:.4g}; estimated cost {best['total_cost']:.2f}. Do not report this tuned result as independent test performance.")
        result=result.copy()
        result["alert"]=(scores>=chosen_threshold).astype(int)
    else:
        st.info("Upload a binary fraud-label column to enable precision-recall and cost analysis. Unlabeled anomaly scores are not fraud probabilities.")
    st.subheader("Anomaly score distribution")
    bins=pd.cut(result["anomaly_score"],bins=20,duplicates="drop")
    counts=result.groupby(bins,observed=True).size()
    counts.index=counts.index.astype(str)
    st.bar_chart(counts)
    if label is not None and label in result.columns:
        chart_data=result[["anomaly_score",label]].copy()
        chart_data["group"]=chart_data[label].map({0:"Labeled legitimate",1:"Labeled fraud"})
        st.scatter_chart(chart_data,x="anomaly_score",y=label,color="group")
    st.subheader("Flagged transactions")
    flagged=result[result["alert"]==1].sort_values("anomaly_score",ascending=False)
    st.dataframe(flagged.head(250),width='stretch')
    st.download_button("Download scored test transactions",result.to_csv(index=False).encode(),
                       file_name="scored_transactions.csv",mime="text/csv")
    if len(flagged):
        st.subheader("Feature deviation inspection")
        st.caption("Descriptive deviations from the held-out population, NOT model feature contributions or causal explanations.")
        position=st.selectbox("Flagged row index",list(flagged.index[:100]))
        deviation=feature_deviations(result,used_features,position)
        st.dataframe(deviation,width='stretch')
        st.caption("Feature deviations are NOT causal explanations or exact model contributions. Score-level perturbation explanations require detector-specific modeling and are planned for a later release.")

