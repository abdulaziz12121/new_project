from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import streamlit as st

st.title("Machine Learning Prediction")

