from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from waitress import serve

app = Flask(__name__)

# Load the trained model and preprocessor from the 'models' folder
model = joblib.load('models/crop_yield_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    temperature = float(data['temperature'])
    rainfall = float(data['rainfall'])
    area = float(data['area'])
    crop_type = data['crop_type']
    soil = data['soil']
    pesticides = data['pesticides']
    fertilizers = float(data['fertilizers'])
    
    # Prepare the input data for the model
    df = pd.DataFrame([[temperature, rainfall, area, crop_type, soil, pesticides, fertilizers]], 
                      columns=['temperature', 'rainfall', 'area', 'crop_type', 'soil', 'pesticides', 'fertilizers'])
    
    # Preprocess the input
    df_transformed = preprocessor.transform(df)
    
    # Make prediction
    prediction = model.predict(df_transformed)[0]
    
    return render_template('index.html', prediction_text=f'Predicted Crop Yield: {prediction:.2f} kg/ha')

if __name__ == "__main__":
    # Use Waitress to serve the app
    serve(app, host='0.0.0.0', port=8000)
