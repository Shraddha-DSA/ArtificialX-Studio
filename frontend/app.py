import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ArtificialX Studio", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
    /* Global Text and Background */
    .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: monospace;
    }
    /* Buttons */
    .stButton>button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Alerts and Info boxes */
    .stAlert {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border-left: 4px solid #000000 !important;
    }
    /* Expander headers */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #f9f9f9 !important;
        border: 1px solid #000000 !important;
    }
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid black;
    }
    /* Dividers */
    hr {
        border-top: 2px solid #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("ArtificialX Studio")
st.markdown("**Train Your Model**")


step = st.sidebar.radio("Navigation", [
    "1. Upload & EDA", 
    "2. Configure & Train", 
    "3. Evaluate", 
    "4. Live Prediction"
])

state_keys = ['filename', 'columns', 'dtypes', 'categorical_uniques', 'corr_matrix', 'missing_counts', 'head', 'train_results', 'model_id', 'task_type']
for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = None

if step == "1. Upload & EDA":
    st.header("Step 1: Upload Dataset & Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Upload to Server"):
            with st.spinner("Analyzing dataset..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    res = requests.post(f"{API_URL}/api/dataset/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.filename = data["filename"]
                        st.session_state.columns = data["columns"]
                        st.session_state.dtypes = data["dtypes"]
                        st.session_state.categorical_uniques = data["categorical_uniques"]
                        st.session_state.corr_matrix = data["correlation_matrix"]
                        st.session_state.missing_counts = data.get("missing_counts", {})
                        st.session_state.head = data.get("head", [])
                        st.success("File uploaded and analyzed successfully!")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error("Cannot connect to backend. Is FastAPI running?")

    if st.session_state.filename:
        st.divider()
        st.subheader("📊 Exploratory Data Analysis (EDA)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Preview:**")
            st.dataframe(pd.DataFrame(st.session_state.head), use_container_width=True)
            
            if st.session_state.missing_counts:
                st.write("**Missing Values:**")
                st.dataframe(pd.DataFrame([st.session_state.missing_counts]).T.rename(columns={0: "Missing Count"}), use_container_width=True)
            else:
                st.write("**Missing Values:** None detected.")
            
        with col2:
            st.write("**Numeric Correlation Heatmap:**")
            corr = st.session_state.corr_matrix
            if corr:
                corr_df = pd.DataFrame(corr)
                fig = px.imshow(corr_df, text_auto=True, color_continuous_scale='gray', aspect="auto")
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found for correlation heatmap.")

elif step == "2. Configure & Train":
    st.header("Step 2: Advanced Configuration & Training")
    
    if not st.session_state.filename:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Target Column (What to predict)", st.session_state.columns)
            task_type = st.radio("Task Type", ["Classification", "Regression"])
        with c2:
            feature_cols = st.multiselect("Feature Columns (Inputs)", [c for c in st.session_state.columns if c != target_col])
            
            clf_models = ["Random Forest Classifier", "Logistic Regression", "Support Vector Machine (SVC)", "K-Nearest Neighbors (KNN)", "Gradient Boosting Classifier", "Decision Tree Classifier"]
            reg_models = ["Random Forest Regressor", "Linear Regression", "Ridge Regression", "Support Vector Regressor (SVR)", "Gradient Boosting Regressor"]
            model_type = st.selectbox("Model Algorithm", clf_models if task_type == "Classification" else reg_models)

        st.markdown("### ⚙️ Advanced Settings")
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            with st.expander("Data Preprocessing"):
                test_size = st.slider("Test Data Split %", 10, 50, 20, 5) / 100.0
                missing_strategy = st.selectbox("Handle Missing Values", ["Mean/Mode (Auto)", "Median/Mode", "Drop Rows"])
                
        with adv_col2:
            with st.expander("Hyperparameter Tuning"):
                hyperparams = {}
                if "Random Forest" in model_type or "Gradient Boosting" in model_type:
                    hyperparams['n_estimators'] = st.slider("Number of Trees", 10, 300, 100, 10)
                    hyperparams['max_depth'] = st.slider("Max Depth", 1, 50, 10)
                elif "KNN" in model_type:
                    hyperparams['n_neighbors'] = st.slider("Number of Neighbors (K)", 1, 20, 5)
                elif "SVC" in model_type or "SVR" in model_type:
                    hyperparams['C'] = st.number_input("Regularization (C)", 0.1, 10.0, 1.0)
                elif "Ridge" in model_type:
                    hyperparams['alpha'] = st.number_input("Alpha", 0.1, 10.0, 1.0)
                else:
                    st.info("No hyperparameters to tune for this specific model.")
        
        if st.button("START TRAINING PROCESS"):
            if not feature_cols:
                st.error("Please select at least one feature.")
            else:
                with st.spinner("Training model on the backend..."):
                    payload = {
                        "filename": st.session_state.filename,
                        "target_col": target_col,
                        "features": feature_cols,
                        "model_type": model_type,
                        "task_type": task_type,
                        "test_size": test_size,
                        "missing_strategy": missing_strategy,
                        "hyperparameters": hyperparams
                    }
                    try:
                        res = requests.post(f"{API_URL}/api/model/train/tabular", json=payload)
                        if res.status_code == 200:
                            st.session_state.train_results = res.json()
                            st.session_state.task_type = task_type
                            st.session_state.model_id = res.json()['model_id']
                            st.success(f"Training Complete! Model ID: {st.session_state.model_id}. Go to Step 3.")
                        else:
                            st.error(f"Training failed: {res.text}")
                    except Exception as e:
                        st.error("Cannot connect to backend.")

elif step == "3. Evaluate":
    st.header("Step 3: Evaluate & Visualizations")
    
    if st.session_state.train_results is None:
        st.warning("Please train a model in Step 2 first.")
    else:
        results = st.session_state.train_results
        metrics = results["metrics"]
        charts = results.get("chart_data", {})
        
        st.subheader("📈 Performance Metrics")
        m1, m2 = st.columns(2)
        if st.session_state.task_type == "Classification":
            m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            

            def bg_gray(s):
                return ['background-color: #e0e0e0; color: black' for v in s]
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.apply(bg_gray))
        else:
            m1.metric("R² Score", f"{metrics['r2']:.4f}")
            m2.metric("Mean Squared Error", f"{metrics['mse']:.4f}")

        st.divider()
        st.subheader("Model Visualizations")
        c1, c2 = st.columns(2)
        
        with c1:
            if 'feature_importances' in charts:
                st.write("**Feature Importances**")
                fi_df = pd.DataFrame(charts['feature_importances']).sort_values(by='importances', ascending=True)
                fig = px.bar(fi_df, x='importances', y='features', orientation='h', color_discrete_sequence=['black'])
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not supported by this algorithm.")
                
        with c2:
            if st.session_state.task_type == "Classification" and 'confusion_matrix' in charts:
                st.write("**Confusion Matrix**")
                z = charts['confusion_matrix']['matrix']
                labels = charts['confusion_matrix']['labels']
                fig = px.imshow(z, text_auto=True, x=labels, y=labels, labels=dict(x="Predicted", y="Actual"), color_continuous_scale='gray')
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
                st.plotly_chart(fig, use_container_width=True)
            elif st.session_state.task_type == "Regression" and 'scatter' in charts:
                st.write("**Actual vs Predicted (Test Set)**")
                scatter_df = pd.DataFrame(charts['scatter'])
                fig = px.scatter(scatter_df, x='actual', y='predicted', opacity=0.6, color_discrete_sequence=['black'])
                fig.add_shape(type="line", x0=scatter_df['actual'].min(), y0=scatter_df['actual'].min(), x1=scatter_df['actual'].max(), y1=scatter_df['actual'].max(), line=dict(color="gray", dash="dash"))
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("💾 Export Model")
        download_url = f"{API_URL}/api/model/download/{st.session_state.model_id}"
        st.markdown(f"**[DOWNLOAD MODEL PACKAGE (.pkl)]({download_url})**")


elif step == "4. Live Prediction":
    st.header("Step 4: Live Inference Engine")
    
    if st.session_state.model_id is None:
        st.warning("Please train a model in Step 2 first to enable live predictions.")
    else:
        st.write("Enter values below to test your model in real-time.")
        
        trained_features = st.session_state.train_results['chart_data'].get('feature_importances', {}).get('features', [])
        if not trained_features:
            trained_features = [col for col in st.session_state.columns if col in st.session_state.dtypes]
            
        with st.form("prediction_form"):
            input_data = {}
            cols = st.columns(2)
            
            for i, feature in enumerate(trained_features):
                dtype = st.session_state.dtypes.get(feature, "object")
                
                with cols[i % 2]:
                    if dtype == "object":
                        options = st.session_state.categorical_uniques.get(feature, ["Unknown"])
                        input_data[feature] = st.selectbox(f"{feature}", options)
                    else:
                        input_data[feature] = st.number_input(f"{feature} (numeric)", value=0.0)
            
            submitted = st.form_submit_button("EXECUTE PREDICTION")
            
            if submitted:
                with st.spinner("Predicting..."):
                    payload = {"model_id": st.session_state.model_id, "input_data": input_data}
                    try:
                        res = requests.post(f"{API_URL}/api/model/predict", json=payload)
                        if res.status_code == 200:
                            prediction = res.json()['prediction']
                            st.success(f"PREDICTION RESULT: {prediction}")
                        else:
                            st.error(f"Prediction failed: {res.text}")
                    except Exception as e:
                        st.error("Cannot connect to backend.")