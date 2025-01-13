from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    'totalcylinderoilspecificconsumption', 'watertemperature', 'winddirection', 'winddirectionisvariable',
    'etanextport', 'reporttypeidcode', 'tugsused', 'utctime', 'voyagenumber', 'distanceeosptofwe',
    'estimatedtimeofdeparture', 'finishedwithenginetime', 'timesteamed', 'bendingmomentsinpercent',
    'dischargedsludge', 'estimatedbunkersnextport', 'estimatedtimeofarrival', 'metacentricheight',
    'shearforcesinpercent', 'standbyenginetime', 'distancetoeosp', 'saileddistance', 'runninghourscountervalue',
    'energyproducedcountervalue', 'energyproducedinreportperiod', 'consumption', 'runninghours',
    'new_fromportcode', 'new_toportcode', 'weather', 'new_timezoneinfo_05:30', 'new_timezoneinfo_07:30',
    'new_timezoneinfo_08:30', 'new_timezoneinfo_09:30', 'new_timezoneinfo_10:30', 'new_timezoneinfo_11:00',
    'new_timezoneinfo_11:30', 'new_timezoneinfo_12:00', 'new_timezoneinfo_12:30', 'new_timezoneinfo_13:30'
]

def estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption,totalcylinderoilspecificconsumption, saileddistance):
    x = np.zeros(len(EXPECTED_FEATURES))
    x[EXPECTED_FEATURES.index('airpressure')] = airpressure
    x[EXPECTED_FEATURES.index('consumption')] = consumption
    x[EXPECTED_FEATURES.index('totalcylinderoilconsumption')] = totalcylinderoilconsumption
    x[EXPECTED_FEATURES.index('totalcylinderoilspecificconsumption')] = totalcylinderoilspecificconsumption
    x[EXPECTED_FEATURES.index('saileddistance')] = saileddistance

    total_consumption = __model.predict([x])[0]
    fuel_per_nautical_mile = total_consumption / saileddistance
    return total_consumption, fuel_per_nautical_mile

def load_saved_artifacts():
    global __data_columns
    global __model
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts\\columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __data_columns = [col for col in __data_columns if col in EXPECTED_FEATURES][:65]
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts\\Decision Tree_best_model.pkl", 'rb') as model_file:
        __model = pickle.load(model_file)
    # Verify the number of features
    if len(__data_columns) != __model.n_features_in_:
        raise ValueError(f"Model expects {__model.n_features_in_} features, but received {len(__data_columns)} features.")

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    # Render a simple HTML form
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fuel Consumption Prediction</title>
    </head>
    <body>
        <h1>Fuel Consumption Prediction</h1>
        <h2>Default Predictions</h2>
        <ul>
            <li>Input: (1011, 70, 100, 0.6,23.80566) → Total Consumption: {{ pred1[0] }} | Fuel/Nautical Mile: {{ pred1[1] }}</li>
            <li>Input: (1000, 70, 80, 0.4,26.734) → Total Consumption: {{ pred2[0] }} | Fuel/Nautical Mile: {{ pred2[1] }}</li>
        </ul>
        <h2>Enter Your Inputs</h2>
        <form action="/predict" method="POST">
            <label for="airpressure">Air Pressure:</label>
            <input type="number" step="0.01" id="airpressure" name="airpressure" required><br><br>
            <label for="consumption">Consumption:</label>
            <input type="number" step="0.01" id="consumption" name="consumption" required><br><br>
            <label for="totalcylinderoilconsumption">Total Cylinder Oil Consumption:</label>
            <input type="number" step="0.01" id="totalcylinderoilconsumption" name="totalcylinderoilconsumption" required><br><br>
            <label for="totalcylinderoilspecificconsumption">Total Cylinder Oil Specific Consumption:</label>
            <input type="number" step="0.01" id="totalcylinderoilspecificconsumption" name="totalcylinderoilspecificconsumption" required><br><br>
            <label for="saileddistance">Sailed Distance:</label>
            <input type="number" step="0.01" id="saileddistance" name="saileddistance" required><br><br>
            <label for="actual_total_consumption">Actual Total Consumption:</label>
            <input type="number" step="0.01" id="actual_total_consumption" name="actual_total_consumption" required><br><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """,
    pred1=estimate_fuel_consumption(1011, 70, 100, 0.6,saileddistance=23.80566),
    pred2=estimate_fuel_consumption(1000, 70, 80, 0.4,saileddistance=2673.4))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    airpressure = float(data['airpressure'])
    consumption = float(data['consumption'])
    totalcylinderoilconsumption = float(data['totalcylinderoilconsumption'])
    totalcylinderoilspecificconsumption = float(data['totalcylinderoilspecificconsumption'])
    saileddistance = float(data['saileddistance'])
    actual_total_consumption = float(data['actual_total_consumption'])

    total_consumption, fuel_per_nautical_mile = estimate_fuel_consumption(
        airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance
    )

    # Calculate accuracy metrics
    mse = mean_squared_error([actual_total_consumption], [total_consumption])
    mae = mean_absolute_error([actual_total_consumption], [total_consumption])

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fuel Consumption Prediction</title>
    </head>
    <body>
        <h1>Prediction Result</h1>
        <p>Total Fuel Consumption: {{ total_consumption }}</p>
        <p>Fuel Consumption per Nautical Mile: {{ fuel_per_nautical_mile }}</p>
        <h2>Accuracy Metrics</h2>
        <p>Mean Squared Error (MSE): {{ mse }}</p>
        <p>Mean Absolute Error (MAE): {{ mae }}</p>
        <a href="/">Go Back</a>
    </body>
    </html>
    """, total_consumption=total_consumption, fuel_per_nautical_mile=fuel_per_nautical_mile, mse=mse, mae=mae)