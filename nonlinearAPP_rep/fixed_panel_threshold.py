
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# Load the data
data = pd.read_csv("panel_data.csv")

# Set the index as the individual identifier and the time dimension
data.set_index(['id', 'time'], inplace=True)

# Separate the dependent variable (y) and independent variables (X)
y = data['dependent_var']
X = data.drop('dependent_var', axis=1)

threshold = 0.5  # Define a threshold value
X['threshold_dummy'] = (X['threshold_var'] > threshold).astype(int)

# Create a Fixed-Effect Panel Model
model = PanelOLS(y, X, entity_effects=True)

# Fit the model to the data
results = model.fit()

# Print the summary of the results
print(results.summary())

# Get the estimated coefficients
coefficients = results.params
print("Coefficients:", coefficients)

# Get the Fixed-Effects estimates
fixed_effects = results.estimated_effects
print("Fixed Effects:", fixed_effects)
