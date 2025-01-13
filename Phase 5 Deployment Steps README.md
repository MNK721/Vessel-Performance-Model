                                     Deployment Steps in Pycharm


Steps to Run the Code Locally
1. Clone the Repository
Clone the project repository from the source (e.g., GitHub):
git clone <repository_url>
cd <repository_folder>
2. Install Dependencies
pip install -r requirements.txt
Example requirements.txt:
pandas
numpy
scikit-learn
xgboost
flask
gunicorn
matplotlib
shap
3. Execute Model Training
Run the script to preprocess the data, train the model, and save the output:
python train_model.py
Input: Historical data (e.g., JSON or CSV files) in the data/ directory.
Output: A trained model saved as models/fuel_model.pkl and visualizations saved in the outputs/ folder.
4. Run the Prediction Script
Use the provided script to make predictions on new data:

bash
Copy code
python predict.py --input data/new_data.json
Input: A JSON file with new data for predictions (example format below).
Output: Predictions saved as predictions/prediction_results.json.
Sample Input (new_data.json):

json
Copy code
[
  {
    "speed": 13.5,
    "distance_traveled": 250,
    "cargo_weight": 1200,
    "sea_state": 4,
    "fuel_type": "MGO"
  }
]
5. Run the API Locally
Start the Flask API for serving predictions:

bash
Copy code
python app.py
The API will be accessible at http://127.0.0.1:5000/predict.
Calling the Endpoint or Script for Predictions
Using API
Send a POST request with input data in JSON format:

bash
Copy code
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '[
           {
             "speed": 12.5,
             "distance_traveled": 200,
             "cargo_weight": 1000,
             "sea_state": 3,
             "fuel_type": "MGO"
           }
         ]'
Response: Predicted fuel consumption.
Using Prediction Script
Run predict.py directly for local predictions:

bash
Copy code
python predict.py --input data/new_data.json
Demonstration Notebook
A Jupyter Notebook (demonstration.ipynb) provides a detailed walkthrough of:

Preprocessing New Data:
Loading and preparing input features.

Making Predictions:
Using the trained model to generate results for new input data.

Visualizing Anomaly Results:
Highlighting outliers or deviations in predicted values.

Example Notebook Cell:
python
Copy code
from predict import predict_fuel

# Input Data
input_data = [
    {"speed": 12.5, "distance_traveled": 200, "cargo_weight": 1000, "sea_state": 3, "fuel_type": "MGO"}
]

# Generate Predictions
prediction = predict_fuel(input_data)

# Display Results
print(f"Predicted Fuel Consumption: {prediction[0]} liters/day")
Folder Structure
kotlin
Copy code
project/
├── data/
│   ├── historical_data.json
│   ├── new_data.json
├── models/
│   └── fuel_model.pkl
├── outputs/
│   └── feature_importance.png
├── predictions/
│   └── prediction_results.json
├── app.py
├── train_model.py
├── predict.py
├── demonstration.ipynb
├── requirements.txt
├── Dockerfile
└── README.md
