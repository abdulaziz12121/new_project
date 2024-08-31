from fastapi import FastAPI, HTTPException,Query
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # يسمح بالوصول من أي مصدر. قم بتقييد هذا في الإنتاج
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

model = joblib.load('DBSCAN_model.joblib')

# model=pickle.load(open('train_model.sav','rb'))

app = FastAPI()


@app.get("/predict")
async def predict(
    yellow: float = Query(..., description="عمر الطالب (من 15 إلى 18 سنة)"),
    red: float = Query(..., description="جنس الطالب (0: ذكر، 1: أنثى)"),
    position_encoded: int = Query(..., description="العرق (0: قوقازي، 1: أفريقي أمريكي، 2: آسيوي، 3: آخر)"),
  
):

 data = {
      "yellow": yellow,
      "red": red,
      "position_encoded": position_encoded
}

 df = pd.DataFrame([data])

 predictions =  model.predict(model, data=df)
  

  
 return predictions


# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import streamlit as st

# # Load the trained model
# model = joblib.load("DBSCAN_model.joblib")

# # Streamlit app code
# st.write("Hello")

# # Get user input values using Streamlit widgets
# yellow = st.number_input('Enter the value for yellow:', min_value=0.0)
# red = st.number_input('Enter the value for red:', min_value=0.0)
# position_encoded = st.number_input('Enter the value for position_encoded:', step=1)

# # FastAPI app instance


# # Pydantic model for input data validation
# class InputFeatures(BaseModel):
#     yellow: float
#     red: float
#     position_encoded: int

# # Endpoint for model prediction
# @app.post("/predict")
# async def predict(data: InputFeatures):
#     features = np.array([data.yellow, data.red, data.position_encoded]).reshape(1, -1)
#     prediction = model.predict(features)
#     return {"prediction": prediction.tolist()}

