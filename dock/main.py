from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import joblib
import numpy as np

# main application object
app = FastAPI(title ="California Housing Price Prediction API")
# loaded the pre-trained model into memory
model = joblib.load("california_housing_model.joblib")

#defines exact shape and data types of the input
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
               "MedInc" : 8.3253,
               "HouseAge" : 41.0,
               "AveRooms" : 6.9841,
               "AveBedrms" : 1.0238,
               "Population" : 322.0,
               "AveOccup": 3.21,
               "Latitude":37.77,
               "Longitude":-122.42
            }
        }
    )

#prediction endpoint, listens to PO             ST requests at that URL
@app.post("/predict", tags=["Predictions"])
async def predict_price(features: HouseFeatures):
    """
    Predicts the median house value(target variable) based on input features
    """
    # Convert the input data into a Numpy array
    # the order of features must be the same as the training
    input_data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
]])
    # the output is a price in units of 100,000 USD
    prediction = model.predict(input_data)[0]

    return {"predicted_median_house_value":f"${prediction * 100000:.2f}"}

@app.get("/",tags=["General"])
async def read_root():
    return {"message":"Welcome to the Housing Price Prediction API!"}