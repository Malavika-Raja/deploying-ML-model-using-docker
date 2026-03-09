import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_model():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y =pd.Series(housing.target, name="MedHouseVal")

    print("Dataset shape:", X.shape)
    print("\nMissing values:\n", X.isnull().sum())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=64)

    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])
    #train the model
    pipeline.fit(X_train, y_train)
    #Predict the model
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    model_filename ='california_housing_model.joblib'
    joblib.dump(pipeline,model_filename)

    print(f"Model saved as {model_filename}.")

if __name__ == "__main__":
    train_and_save_model()