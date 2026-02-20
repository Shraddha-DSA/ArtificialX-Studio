from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import pandas as pd
from typing import List, Dict, Any
from ml_utils import train_tabular_model, predict_from_model

app = FastAPI(title="Teachable Machine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TrainRequest(BaseModel):
    filename: str
    target_col: str
    features: List[str]
    model_type: str
    task_type: str
    test_size: float
    missing_strategy: str
    hyperparameters: Dict[str, Any]

class PredictRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        df = pd.read_csv(file_path)
        
        
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        categorical_uniques = {
            col: df[col].dropna().astype(str).unique().tolist() 
            for col in df.columns if df[col].dtype == 'object'
        }
        
        
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr().fillna(0).to_dict() if not numeric_df.empty else {}

        return {
            "filename": file.filename, 
            "columns": columns,
            "dtypes": dtypes,
            "categorical_uniques": categorical_uniques,
            "correlation_matrix": corr_matrix,
            "missing_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "head": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/train/tabular")
async def train_model(request: TrainRequest):
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    try:
        result = train_tabular_model(
            file_path=file_path,
            target_col=request.target_col,
            features=request.features,
            model_type=request.model_type,
            task_type=request.task_type,
            test_size=request.test_size,
            missing_strategy=request.missing_strategy,
            hyperparameters=request.hyperparameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/predict")
async def predict_live(request: PredictRequest):
    
    try:
        prediction = predict_from_model(request.model_id, request.input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/download/{model_id}")
async def download_model(model_id: str):
    model_path = os.path.join(UPLOAD_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")
    return FileResponse(path=model_path, filename=f"model_{model_id}.pkl")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)