# Importing libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Loading scraped data
df = pd.read_csv('Flight_data.csv')

# Dropping duplicate tuples
df.drop_duplicates(keep = 'first', inplace = True)

# Removing remaining empty fields
df.dropna(axis=0, inplace=True)

# Exporting cleaned data
df.to_csv('Cleaned_data.csv')