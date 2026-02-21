# ⬛ ArtificialX Studio  
### Enterprise Machine Learning Workspace

ArtificialX Studio is a **containerized AutoML platform** that enables users to train, evaluate, and deploy machine learning models on tabular data through an intuitive web interface.

---
**Detailed Blog** 

https://medium.com/@shraddhatiwari345/how-i-built-artificialx-studio-ad2eef7fa659

---
## ✨ Key Features

- 📂 Upload CSV datasets  
- 📊 Automated EDA & profiling  
- 🤖 Classification & Regression models  
- ⚙️ Built-in preprocessing & encoding  
- 🎛 Hyperparameter tuning  
- 📈 Interactive metrics & visualizations  
- 💾 Download trained models (`.pkl`)  
- 🚀 Live browser-based inference  

---

## 🔄 4-Phase Workflow

### 1️⃣ Data Ingestion & Profiling
- Automatic missing value detection  
- Data type inference  
- Correlation heatmaps  

### 2️⃣ Model Configuration
- Select algorithm (RF, SVM, KNN, Logistic, Linear, SVR, etc.)  
- Preprocessing options  
- Hyperparameter customization  

### 3️⃣ Performance Analytics
- Accuracy, R², MSE  
- Confusion Matrix  
- Feature Importance  
- Actual vs Predicted plots  

### 4️⃣ Live Inference
- Dynamic feature input form  
- Real-time predictions via FastAPI  

---

## 🏗️ Architecture

- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **ML Engine:** Scikit-Learn  
- **Infrastructure:** Docker & Docker Compose  

```
artificialx-studio/
├── docker-compose.yml
├── backend/
│   ├── main.py
│   ├── ml_utils.py
│   └── requirements.txt
└── frontend/
    ├── app.py
    └── requirements.txt
```

---

## 🚀 Quick Start (Docker)

```bash
git clone https://github.com/YOUR_USERNAME/artificialx-studio.git
cd artificialx-studio
docker-compose up --build
```

- 🌐 Frontend → http://localhost:8501  
- 📘 Backend Docs → http://localhost:8000/docs  

---

## 🛠️ Run Locally (Without Docker)

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Using Downloaded Models

```python
import joblib
import pandas as pd

artifact = joblib.load("model.pkl")
model = artifact["model"]
features = artifact["features"]

data = pd.DataFrame([{"Feature1": 10, "Feature2": "A"}])
prediction = model.predict(data[features])
print(prediction)
```

---

## 📄 License

This project is licensed under the **MIT License**.
