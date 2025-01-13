from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

__data_columns = None
__model = None

# List of 65 features used by the Decision Tree Regressor
EXPECTED_FEATURES = [
    'airpressure', 'airtemperature', 'averagespeedgps', 'averagespeedlog', 'cargometrictons', 'currentstrength',
    'distancefromlastport', 'distancetonextport', 'distancetravelledsincelastreport', 'enginedriftingstoppagetime',
    'engineroomairpressure', 'engineroomairtemperature', 'engineroomrelativeairhumidity', 'engineslip',
    'isfuelchangeover', 'isturbochargercutout', 'relativeairhumidity', 'remainingdistancetoeosp', 'remainingtimetoeosp',
    'scavengingaircoolingwatertemperatureaftercooler', 'scavengingairpressure', 'scavengingairtemperatureaftercooler',
    'seastate', 'seastatedirection', 'totalcylinderoilconsumption', 'totalcylinderoilspecificconsumption',
    'watertemperature', 'winddirection', 'winddirectionisvariable', 'etanextport', 'reporttypeidcode', 'tugsused',
    'utctime', 'voyagenumber', 'distanceeosptofwe', 'estimatedtimeofdeparture', 'finishedwithenginetime', 'timesteamed',
    'bendingmomentsinpercent', 'dischargedsludge', 'estimatedbunkersnextport', 'estimatedtimeofarrival',
    'metacentricheight', 'shearforcesinpercent', 'standbyenginetime', 'distancetoeosp', 'saileddistance',
    'runninghourscountervalue', 'energyproducedcountervalue', 'energyproducedinreportperiod', 'consumption', 'runninghours',
    'new_fromportcode', 'new_toportcode', 'weather', 'new_timezoneinfo_05:30', 'new_timezoneinfo_07:30',
    'new_timezoneinfo_08:30', 'new_timezoneinfo_09:30', 'new_timezoneinfo_10:30', 'new_timezoneinfo_11:00',
    'new_timezoneinfo_11:30', 'new_timezoneinfo_12:00', 'new_timezoneinfo_12:30', 'new_timezoneinfo_13:30'
]

def estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, saileddistance):
    x = np.zeros(len(EXPECTED_FEATURES))
    x[EXPECTED_FEATURES.index('airpressure')] = airpressure
    x[EXPECTED_FEATURES.index('consumption')] = consumption
    x[EXPECTED_FEATURES.index('totalcylinderoilconsumption')] = totalcylinderoilconsumption
    x[EXPECTED_FEATURES.index('saileddistance')] = saileddistance

    total_consumption = __model.predict([x])[0]

    # Calculate fuel consumption per nautical mile
    fuel_per_nautical_mile = total_consumption / saileddistance

    return fuel_per_nautical_mile

def load_saved_artifacts():
    global __data_columns
    global __model

    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts\\columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']

    # Filter `__data_columns` to match `EXPECTED_FEATURES` and remove extra features
    __data_columns = [col for col in __data_columns if col in EXPECTED_FEATURES][:65]

    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server\\artifacts\\Decision Tree_best_model.pkl", 'rb') as model_file:
        __model = pickle.load(model_file)

    # Verify the number of features
    if len(__data_columns) != __model.n_features_in_:
        raise ValueError(f"Model expects {__model.n_features_in_} features, but received {len(__data_columns)} features.")

    print("Artifacts loaded successfully")
    print(f"Common features: {__data_columns}")

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fuel Consumption Prediction API! The server is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    airpressure = data['airpressure']
    consumption = data['consumption']
    totalcylinderoilconsumption = data['totalcylinderoilconsumption']
    saileddistance = data['saileddistance']
    response = {
        'fuel_per_nautical_mile': estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, saileddistance)
    }
    return jsonify(response)

if __name__ == "__main__":
    load_saved_artifacts()
    print(f"Expected features: {EXPECTED_FEATURES}")
    print(f"Model expects {__model.n_features_in_} features")
    print("Predicted values:")
    print(estimate_fuel_consumption(1016, 70, 100, 23.80556))
    print(estimate_fuel_consumption(1000, 70, 80, 29.80556))
    app.run(debug=True)










