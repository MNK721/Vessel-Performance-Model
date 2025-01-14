                                                       Vessel Performance Model
                                                        
                                                     Deployment Steps in Pycharm


Steps to Run the Code Locally

1. Clone the Repository

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

python train_model.py
Input: Historical data /cleaned data(CSV Format)
Output: A trained model saved as .pkl files (eg:Decision Tree_best_model.pkl) and visualizations saved in the outputs folders.

4. Run the Prediction Script
Use the provided script to make predictions on new data:

python predict.py --cleaned data final.json
Input: A JSON file with new data for predictions (example format below).
Output: Predictions saved as predictions/prediction_results.json.


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



python app.py
The API will be accessible at http://127.0.0.1:5000


Git commands::


git pull origin master 
git add .
git push origin master 
git commit -m "Modified flask API to predict total consumption and fuel per nautical mile"
git push origin master 


Calling the Endpoint or Script for Predictions
Using API
Send a POST request with input data in JSON format:


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


python predict.py cleaned data final.json


from predict import predict_fuel

# Input Data
input_data = [
    {"speed": 12.5, "distance_traveled": 200, "cargo_weight": 1000, "sea_state": 3, "fuel_type": "MGO"}
]

# Generate Predictions
prediction = predict_fuel(input_data)

# Display Results
print(f"Predicted Fuel Consumption: {prediction[0]} liters/day")

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


                                                                 Flask Server App Code



from flask import Flask, request, render_template_string
import pickle
import numpy as np
import json

__data_columns = None
__model = None

# List of 65 features used by the Decision Tree Regressor
EXPECTED_FEATURES = [
    'airpressure', 'airtemperature', 'averagespeedgps', 'averagespeedlog', 'cargometrictons',
    'currentstrength', 'distancefromlastport', 'distancetonextport', 'distancetravelledsincelastreport',
    'enginedriftingstoppagetime', 'engineroomairpressure', 'engineroomairtemperature',
    'engineroomrelativeairhumidity', 'engineslip', 'isfuelchangeover', 'isturbochargercutout',
    'relativeairhumidity', 'remainingdistancetoeosp', 'remainingtimetoeosp',
    'scavengingaircoolingwatertemperatureaftercooler', 'scavengingairpressure',
    'scavengingairtemperatureaftercooler', 'seastate', 'seastatedirection', 'totalcylinderoilconsumption',
    'totalcylinderoilspecificconsumption', 'watertemperature', 'winddirection',
    'winddirectionisvariable', 'etanextport', 'reporttypeidcode', 'tugsused', 'utctime', 'voyagenumber',
    'distanceeosptofwe', 'estimatedtimeofdeparture', 'finishedwithenginetime', 'timesteamed',
    'bendingmomentsinpercent', 'dischargedsludge', 'estimatedbunkersnextport',
    'estimatedtimeofarrival', 'metacentricheight', 'shearforcesinpercent', 'standbyenginetime',
    'distancetoeosp', 'saileddistance', 'runninghourscountervalue', 'energyproducedcountervalue',
    'energyproducedinreportperiod', 'consumption', 'runninghours', 'new_fromportcode',
    'new_toportcode', 'weather', 'new_timezoneinfo_05:30', 'new_timezoneinfo_07:30',
    'new_timezoneinfo_08:30', 'new_timezoneinfo_09:30', 'new_timezoneinfo_10:30',
    'new_timezoneinfo_11:00', 'new_timezoneinfo_11:30', 'new_timezoneinfo_12:00',
    'new_timezoneinfo_12:30', 'new_timezoneinfo_13:30'
]

def load_saved_artifacts():
    global __data_columns
    global __model

    # Load feature columns
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts\\columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
    __data_columns = [col for col in __data_columns if col in EXPECTED_FEATURES][:65]

    # Load the trained model
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts\\Decision Tree_best_model.pkl", 'rb') as model_file:
        __model = pickle.load(model_file)

    # Verify the model and features match
    if len(__data_columns) != __model.n_features_in_:
        raise ValueError(f"Model expects {__model.n_features_in_} features, but received {len(__data_columns)} features.")
    print("Artifacts loaded successfully")
    print(f"Common features: {__data_columns}")

def estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance):
    x = np.zeros(len(EXPECTED_FEATURES))
    x[EXPECTED_FEATURES.index('airpressure')] = airpressure
    x[EXPECTED_FEATURES.index('consumption')] = consumption
    x[EXPECTED_FEATURES.index('totalcylinderoilconsumption')] = totalcylinderoilconsumption
    x[EXPECTED_FEATURES.index('totalcylinderoilspecificconsumption')] = totalcylinderoilspecificconsumption
    x[EXPECTED_FEATURES.index('saileddistance')] = saileddistance

    total_consumption = __model.predict([x])[0]
    fuel_per_nautical_mile = total_consumption / saileddistance
    return total_consumption, fuel_per_nautical_mile

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
        <h1>Fuel Consumption Prediction API</h1>
        <form action="/predict" method="post">
            <label for="airpressure">Air Pressure:</label>
            <input type="text" id="airpressure" name="airpressure"><br><br>
            <label for="consumption">Consumption:</label>
            <input type="text" id="consumption" name="consumption"><br><br>
            <label for="totalcylinderoilconsumption">Total Cylinder Oil Consumption:</label>
            <input type="text" id="totalcylinderoilconsumption" name="totalcylinderoilconsumption"><br><br>
            <label for="totalcylinderoilspecificconsumption">Total Cylinder Oil Specific Consumption:</label>
            <input type="text" id="totalcylinderoilspecificconsumption" name="totalcylinderoilspecificconsumption"><br><br>
            <label for="saileddistance">Sailed Distance:</label>
            <input type="text" id="saileddistance" name="saileddistance"><br><br>
            <input type="submit" value="Predict">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    airpressure = float(request.form['airpressure'])
    consumption = float(request.form['consumption'])
    totalcylinderoilconsumption = float(request.form['totalcylinderoilconsumption'])
    totalcylinderoilspecificconsumption = float(request.form['totalcylinderoilspecificconsumption'])
    saileddistance = float(request.form['saileddistance'])

    total_consumption, fuel_per_nautical_mile = estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance)

    return render_template_string('''
        <h1>Fuel Consumption Prediction</h1>
        <p>Air Pressure: {{ airpressure }}</p>
        <p>Consumption: {{ consumption }}</p>
        <p>Total Cylinder Oil Consumption: {{ totalcylinderoilconsumption }}</p>
        <p>Total Cylinder Oil Specific Consumption: {{ totalcylinderoilspecificconsumption }}</p>
        <p>Sailed Distance: {{ saileddistance }}</p>
        <h2>Total Consumption: {{ total_consumption }}</h2>
        <h2>Fuel Per Nautical Mile: {{ fuel_per_nautical_mile }}</h2>
        <a href="/">Back to Home</a>
    ''', airpressure=airpressure, consumption=consumption, totalcylinderoilconsumption=totalcylinderoilconsumption, totalcylinderoilspecificconsumption=totalcylinderoilspecificconsumption, saileddistance=saileddistance, total_consumption=total_consumption, fuel_per_nautical_mile=fuel_per_nautical_mile)

if __name__ == "__main__":
    try:
        load_saved_artifacts()
        print("Server is running.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        exit(1)
    app.run(debug=True)



