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
    if len(_data_columns) != __model.n_features_in:
        raise ValueError(f"Model expects {_model.n_features_in} features, but received {len(__data_columns)} features.")
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
app = Flask(_name_)

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

if _name_ == "_main_":
    try:
        load_saved_artifacts()
        print("Server is running.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        exit(1)
    app.run(debug=True)


