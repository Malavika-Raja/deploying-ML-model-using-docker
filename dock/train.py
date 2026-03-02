import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import joblib

def train_and_save_model():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y =pd.Series(housing.target, name="MedHouseVal")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=64)

    model = RandomForestRegressor(n_estimators=100,random_state=42)

    model.fit(X_train, y_train)

    model_filename ='california_housing_model.joblib'
    joblib.dump(model,model_filename)

    print(f"Model saved as {model_filename}. Training script is done!")

if __name__ == "__main__":
    train_and_save_model()