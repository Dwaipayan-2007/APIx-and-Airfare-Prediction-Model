# Importing necessary libraries
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Loading price prediction model
def load_model():
    return joblib.load("flight_price_model.pkl")
model = load_model()

# Reading input data
df = pd.read_csv('Cleaned_data.csv')

# Modifying input data as per model
weight_dict = {
    'T4': 1.00, 'T3': 1.11, 'R1': 1.23, 'T1': 1.25, 'Q4': 1.35, 'R3': 1.36, 'R4': 1.39, 'Q3': 1.40, 'T0': 1.42, 'Q2': 1.45, 
    'U2': 1.46, 'R0': 1.46, 'Q0': 1.48, 'U1': 1.49, 'V0': 1.49, 'U0': 1.51, 'U3': 1.56, 'R2': 1.63, 'P2': 1.66, 'T2': 1.72, 
    'O4': 1.87, 'P3': 1.89, 'O3': 1.95, 'O1': 2.22, 'O2': 2.32, 'N2': 2.56}
df['fare_class_weight'] = df['fare_class_code'].map(weight_dict)
df.drop(columns = ['fare_class_code'], inplace = True)
time_delta = pd.to_timedelta(df['departure_time'] + ':00')
df['departure_time'] = (time_delta.dt.total_seconds() // 60).astype(int)
time_delta = pd.to_timedelta(df['arrival_time'] + ':00')
df['arrival_time'] = (time_delta.dt.total_seconds() // 60).astype(int)
data = df[['advance_days', 'departure_time', 'arrival_time', 'duration_minutes',
       'available_seats', 'fare_class_weight']]
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X = poly_transformer.fit_transform(data)

# Predicting fares
total_fare = model.predict(X)

# Exporting the fares
res = pd.DataFrame(total_fare)
res.to_csv('predicted_fare.csv')