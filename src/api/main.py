"""
FastAPI application for NBA over/under prediction.
"""
import os
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path

from src.ml.model import NBAOverUnderModel


# Environment variables
DB_PATH = os.getenv("DB_PATH", "data/db.nba.sqlite")
MODEL_PATH = os.getenv("MODEL_PATH", "models/nba_over20_model.pt")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Initialize FastAPI app
app = FastAPI(title="NBA Over/Under Predictor API")

# Setup templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Global model variable
model = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    player_name: str
    points_line: float
    features: dict


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prob_over: float
    prediction: str
    points_line: float


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model
    
    # Compute project root
    project_root = Path(__file__).parent.parent.parent
    
    # Load model
    model_file = project_root / MODEL_PATH
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    print(f"Loading model from {model_file}...")
    model = NBAOverUnderModel(input_size=6, hidden_size=32)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")
    
    # Check database file
    db_file = project_root / DB_PATH
    if db_file.exists():
        print(f"Database found at {db_file}")
    else:
        print(f"Warning: Database not found at {db_file}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    
    project_root = Path(__file__).parent.parent.parent
    db_file = project_root / DB_PATH
    db_exists = db_file.exists()
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "db_connected": db_exists
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict whether a player will go over or under the points line."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract features in fixed order
    feature_order = [
        'minutes',
        'rebounds',
        'assists',
        'field_goals_attempted',
        'three_pa',
        'free_throws_attempted'
    ]
    
    # Validate all features are present
    missing_features = [feat for feat in feature_order if feat not in request.features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {', '.join(missing_features)}"
        )
    
    # Build feature vector
    feature_vector = [request.features[feat] for feat in feature_order]
    
    # Convert to tensor
    features_tensor = torch.FloatTensor([feature_vector])
    
    # Run model prediction
    with torch.no_grad():
        prob_over = model(features_tensor).item()
    
    # Compute threshold based on points_line
    base_line = 20.0
    line_diff = request.points_line - base_line
    threshold = 0.5 + (line_diff / 100.0)
    threshold = max(0.3, min(0.7, threshold))
    
    # Make prediction
    prediction = "over" if prob_over >= threshold else "under"
    
    return PredictionResponse(
        prob_over=round(prob_over, 4),
        prediction=prediction,
        points_line=request.points_line
    )


@app.get("/example")
async def get_example():
    """Return an example request body for the /predict endpoint."""
    return {
        "player_name": "LeBron James",
        "points_line": 20,
        "features": {
            "minutes": 34.5,
            "rebounds": 8.0,
            "assists": 7.2,
            "field_goals_attempted": 18,
            "three_pa": 6,
            "free_throws_attempted": 6
        }
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
