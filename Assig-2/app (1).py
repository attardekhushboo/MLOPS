import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, filepath):
        data = pd.read_csv(filepath)
        data = data.dropna(subset=['SalePrice'])
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','BedroomAbvGr']
        X = data[features]
        y = data['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)

    def predict(self, features):
        features_df = pd.DataFrame([features])
        scaled_features = self.scaler.transform(features_df)
        prediction = self.model.predict(scaled_features)
        return prediction[0]


price_house = HousePricePredictor()
price_house.load_and_preprocess_data("C:/Users/attar/Downloads/house-prices-advanced-regression-techniques/train.csv")

app = FastAPI()
class HouseFeatures(BaseModel):
    OverallQual:int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int
    BedroomAbvGr: int
predictor = HousePricePredictor()
predictor.load_and_preprocess_data("C:/Users/attar/Downloads/house-prices-advanced-regression-techniques/train.csv")

@app.post("/predict")
def predict_price(features: HouseFeatures):
    features_dict = features.model_dump()
    predicted_price = predictor.predict(features_dict)

    return {"predicted_price": predicted_price}

