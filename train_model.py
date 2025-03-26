import pandas as pd
import numpy as np
import time
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.responses import StreamingResponse
import psutil
import logging
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Pydantic models for Input Validation and Response Structure
class ForecastRequest(BaseModel):
    lagged_values: List[float] = Field(..., min_items=7, max_items=7, description="A list of 7 lagged temperature values")
    location: str = Field(..., description="City and region name (e.g., 'Ajax, Ontario')")
    
    class Config:
        schema_extra = {
            "example": {
                "lagged_values": [15.2, 16.8, 14.1, 13.7, 15.9, 14.5, 16.2],
                "location": "Ajax, Ontario"
            }
        }

class ForecastResponse(BaseModel):
    location: str = Field(..., description="City and region name (e.g., 'Ajax, Ontario')")
    forecast: List[float] = Field(..., description="List of predicted temperature values for the next 7 days")
    response_time_ms: float = Field(..., description="Time taken to process the request, in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "location": "Ajax, Ontario",
                "forecast": [16.5, 16.8, 17.0, 17.3, 17.1, 17.2, 17.4],
                "response_time_ms": 123.45
            }
        }

# Model training and prediction setup
def train_model(data):
    """Train a linear regression model on the weather data."""
    lag = 7
    data = data[['tavg']]  # Using average temperature
    data = create_lag_features(data, lag)
    data.dropna(inplace=True)

    X = data[[f"lag_{i}" for i in range(1, lag + 1)]]
    y = data['tavg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")
    return model

def create_lag_features(data, lag):
    """Create lagged features for supervised learning."""
    for i in range(1, lag + 1):
        data[f"lag_{i}"] = data['tavg'].shift(i)
    return data

# Simulate training with dummy data (replace with actual weather dataset)
dummy_data = pd.DataFrame({"tavg": np.random.uniform(10, 30, size=100)})
model = train_model(dummy_data)

# API Endpoints
@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    start_time = time.time()  # Start timing the request
    process = psutil.Process()  # Memory consumption tracking
    memory_before = process.memory_info().rss / (1024 * 1024)  # Before memory usage (MB)

    # Validate input and predict
    lagged_values = np.array(request.lagged_values).reshape(1, -1)
    predictions = model.predict(lagged_values).flatten()

    memory_after = process.memory_info().rss / (1024 * 1024)  # After memory usage (MB)
    response_time = (time.time() - start_time) * 1000  # Response time (ms)

    return ForecastResponse(
        location=request.location,
        forecast=predictions.tolist(),
        response_time_ms=response_time
    )

@app.get("/forecast/plot")
async def forecast_plot():
    # Example forecast data (replace with predictions)
    predictions = [16.5, 16.8, 17.0, 17.3, 17.1, 17.2, 17.4]
    days = list(range(1, len(predictions) + 1))

    # Create plot
    plt.plot(days, predictions, marker='o')
    plt.title("Weather Forecast")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")

    # Save plot to BytesIO stream
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    
    data = pd.read_csv(file.file)
    # Process data and generate predictions here
    return {"message": "File processed successfully"}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


