import pandas as pd
import numpy as np
import uuid
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix

UPLOAD_DIR = "uploads"

def train_tabular_model(file_path, target_col, features, model_type, task_type, test_size, missing_strategy, hyperparameters):
    df = pd.read_csv(file_path)
    
    if missing_strategy == "Drop Rows":
        df = df.dropna(subset=features + [target_col])
    
    X = df[features].copy()
    y = df[target_col].copy()
 
    if missing_strategy == "Mean/Mode (Auto)":
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
            else:
                X[col] = X[col].fillna(X[col].mean())
    elif missing_strategy == "Median/Mode":
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
            else:
                X[col] = X[col].fillna(X[col].median())

    encoders = {}
    categorical_uniques = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            categorical_uniques[col] = X[col].dropna().unique().tolist()
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
    target_encoder = None
    if task_type == "Classification" and y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    scale_models = ["Logistic Regression", "Linear Regression", "Ridge Regression", "Support Vector Machine (SVC)", "Support Vector Regressor (SVR)", "K-Nearest Neighbors (KNN)"]
    if model_type in scale_models:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    

    if model_type == "Random Forest Classifier":
        model = RandomForestClassifier(**hyperparameters, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Support Vector Machine (SVC)":
        model = SVC(**hyperparameters, probability=True)
    elif model_type == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(**hyperparameters)
    elif model_type == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(**hyperparameters, random_state=42)
    elif model_type == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor(**hyperparameters, random_state=42)
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        model = Ridge(**hyperparameters)
    elif model_type == "Support Vector Regressor (SVR)":
        model = SVR(**hyperparameters)
    elif model_type == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(**hyperparameters, random_state=42)
    else:
        raise ValueError("Unsupported model type")
        
    model.fit(X_train, y_train)
    

    predictions = model.predict(X_test)
    metrics = {}
    chart_data = {}
    
    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_.tolist()
    elif hasattr(model, 'coef_'):
        feature_importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_).tolist()
    
    if feature_importances:
        chart_data['feature_importances'] = {
            'features': features,
            'importances': feature_importances
        }

    if task_type == "Classification":
        metrics['accuracy'] = accuracy_score(y_test, predictions)
        metrics['report'] = classification_report(y_test, predictions, output_dict=True)
        labels = target_encoder.classes_.tolist() if target_encoder else np.unique(y).tolist()
        chart_data['confusion_matrix'] = {
            'matrix': confusion_matrix(y_test, predictions).tolist(),
            'labels': [str(l) for l in labels]
        }
    else:
        metrics['mse'] = mean_squared_error(y_test, predictions)
        metrics['r2'] = r2_score(y_test, predictions)
        limit = min(300, len(y_test)) 
        chart_data['scatter'] = {
            'actual': y_test[:limit].tolist(),
            'predicted': predictions[:limit].tolist()
        }

    model_id = str(uuid.uuid4())[:8]
    model_artifact = {
        'model': model,
        'features': features,
        'encoders': encoders,
        'target_encoder': target_encoder,
        'scaler': scaler if model_type in scale_models else None,
        'task_type': task_type,
        'categorical_uniques': categorical_uniques
    }
    joblib.dump(model_artifact, os.path.join(UPLOAD_DIR, f"{model_id}.pkl"))
    
    return {
        "status": "success",
        "model_id": model_id,
        "metrics": metrics,
        "chart_data": chart_data
    }

def predict_from_model(model_id, input_data):
    model_path = os.path.join(UPLOAD_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise ValueError("Model not found")
        
    artifact = joblib.load(model_path)
    model = artifact['model']
    features = artifact['features']
    
    df_input = pd.DataFrame([input_data])
    
    for col in features:
        if col in artifact['encoders']:
            try:
                df_input[col] = artifact['encoders'][col].transform(df_input[col].astype(str))
            except ValueError:
                df_input[col] = 0 
                
    X_pred = df_input[features]
    if artifact['scaler']:
        X_pred = artifact['scaler'].transform(X_pred)
        
    pred = model.predict(X_pred)[0]
    
    if artifact['target_encoder']:
        pred = artifact['target_encoder'].inverse_transform([pred])[0]
        
    return str(pred) if artifact['task_type'] == "Classification" else float(pred)