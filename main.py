# from fastapi import FastAPI, HTTPException
# import pickle
# from pydantic import BaseModel
# from fastapi import FastAPI
# import uvicorn
# import joblib

# model = joblib.load('DBSCAN_model.joblib')
# scaler = joblib.load('Models/scaler.joblib')

# # model=pickle.load(open('train_model.sav','rb'))

# app = FastAPI()


# class InputFeatures(BaseModel):
#     yellow:float
#     red:float
#     position_encoded:int
    
# def preprocessing(input_features: InputFeatures):
#         dict_f = {
#     'yellow': input_features.yellow,
#         'red': input_features.red,
#     'position_encoded': input_features.position_encoded,

#     }
#         return dict_f

    
# @app.get("/predict")
# def predict(input_features: InputFeatures):
#     return preprocessing(input_features)

# # @app.get("/predict")
# # def predict(input_features: InputFeatures):
# #       dict_f = {
# #     'yellow': input_features.yellow,
# #         'red': input_features.red,
# #     'position_encoded': input_features.position_encoded,

# #     }
# #       predict=model(dict_f)

# @app.post("/predict")
# async def predict(input_features: InputFeatures):
#     data = preprocessing(input_features)
#     y_pred = model.predict(data)
#     return {"pred": y_pred.tolist()[0]}

from fastapi import FastAPI, HTTPException
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib
import numpy as np

# Load the trained model
model = joblib.load('DBSCAN_model.joblib')

# FastAPI app instance
app = FastAPI()

# Pydantic model for input data validation
class InputFeatures(BaseModel):
    yellow: float
    red: float
    position_encoded: int

# Endpoint for model prediction
@app.post("/predict")
async def predict(data: InputFeatures):
    features = np.array([data.yellow, data.red, data.position_encoded]).reshape(1, -1)
    prediction = model.fit_predict(features)
    return {"prediction": prediction.tolist()}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
