import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load dataset
data = pd.read_csv('dataset.csv')

# Separate features and target variable
X = data[['temperature', 'rainfall', 'area', 'crop_type', 'soil', 'pesticides', 'fertilizers']]
y = data['yield']

# Define preprocessing for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['crop_type', 'soil', 'pesticides']),
    ],
    remainder='passthrough'
)

# Preprocess features
X_transformed = preprocessor.fit_transform(X)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_transformed, y)

# Save model and preprocessor in the 'models' folder
joblib.dump(model, 'models/crop_yield_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("Model and preprocessor saved in the 'models' folder.")
