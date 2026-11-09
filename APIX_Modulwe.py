# Importing necessary libraries
import numpy as np
import pandas as pd
import joblib
from scipy.stats import gmean
import warnings
warnings.filterwarnings('ignore')

# Reading fares of a route
df = pd.read_csv('Cleaned_data.csv')

df2 = df[(df['origin'] == 'BLR') & (df['destination'] == 'HYD')]

# Calculating Geometric mean of fares
GM = gmean(df2['total_fare'])

GM2 = 5809 # Geometric mean of base year fares - 2024

# Calculating APIx Formula given by: {(GM of fares for a route) / (GM of base period fare for same route taken 5809)} * 100
APIx = round(GM * 100 / GM2)

# Exporting real-time APIx for the route
apix = pd.DataFrame({'values': [APIx]})
apix.to_csv('APIx_calc.csv')