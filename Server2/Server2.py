# import json
# import pickle
# import numpy as np
# from flask import Flask, request, render_template_string
#
# # Globals
# __data_columns = None
# __model = None
# common_features = None
#
#
# def load_saved_artifacts():
#     global __data_columns
#     global __model
#     global common_features
#
#     # Load feature columns
#     with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server2\\artifacts2\\columns.json", 'r') as f:
#         __data_columns = json.load(f)['data_columns']
#
#     # Load the trained model
#     with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server2\\artifacts2\\Decision Tree_nautical_mile.pkl",
#               'rb') as model_file:
#         __model = pickle.load(model_file)
#
#     # Remove 'fuelpernauticalmile' if it is in the data columns
#     if 'fuelpernauticalmile' in __data_columns:
#         __data_columns.remove('fuelpernauticalmile')
#
#     # Assume model features are correct
#     model_features = __data_columns
#     print("Using columns.json for feature names.")
#
#     # Verify the loaded features
#     common_features = [feature for feature in model_features if feature in __data_columns]
#
#     if len(model_features) != len(common_features):
#         missing_features = [feature for feature in model_features if feature not in common_features]
#         extra_features = [feature for feature in __data_columns if feature not in common_features]
#         print(f"Missing features: {missing_features}")
#         print(f"Extra features: {extra_features}")
#         raise ValueError(f"Model expects {len(model_features)} features, but received {len(common_features)}.")
#
#     print("Artifacts loaded successfully.")
#     print(f"Model"

















import json
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# Globals
__data_columns = None
__model = None
common_features = None

def load_saved_artifacts():
    global __data_columns
    global __model
    global common_features

    # Load feature columns
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server2\\artifacts2\\columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']

    # Load the trained model
    with open("C:\\Users\\mnkmr\\Downloads\\Final Vessel Project\\Server2\\artifacts2\\Decision Tree_nautical_mile.pkl", 'rb') as model_file:
        __model = pickle.load(model_file)

    if 'fuelpernauticalmile' in __data_columns:
        __data_columns.remove('fuelpernauticalmile')

    # Assume model features are correct
    model_features = __data_columns
    print("Using columns.json for feature names.")

    # Verify the loaded features
    common_features = [feature for feature in model_features if feature in __data_columns]


    if len(model_features) != len(common_features):
        missing_features = [feature for feature in model_features if feature not in common_features]
        extra_features = [feature for feature in __data_columns if feature not in common_features]
        print(f"Missing features: {missing_features}")
        print(f"Extra features: {extra_features}")
        raise ValueError(f"Model expects {len(model_features)} features, but received {len(common_features)}.")

    print("Artifacts loaded successfully.")
    print(f"Model features: {model_features}")
    print(f"Common features used: {common_features}")

def estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance):
    x = np.zeros(len(common_features))
    input_features = {
        'airpressure': airpressure,
        'consumption': consumption,
        'totalcylinderoilconsumption': totalcylinderoilconsumption,
        'totalcylinderoilspecificconsumption': totalcylinderoilspecificconsumption,
        'saileddistance': saileddistance
    }

    for feature in common_features:
        if feature in input_features:
            x[common_features.index(feature)] = input_features[feature]

    total_consumption = __model.predict([x])[0]
    fuel_per_nautical_mile = total_consumption / saileddistance if saileddistance != 0 else 0
    return total_consumption, fuel_per_nautical_mile

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fuel Consumption Prediction API</title>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <h1 class="mt-5">Fuel Consumption Prediction API</h1>
                <form action="/predict" method="post" class="mt-3">
                    <div class="form-group">
                        <label for="airpressure">Air Pressure:</label>
                        <input type="text" class="form-control" id="airpressure" name="airpressure" required>
                    </div>
                    <div class="form-group">
                        <label for="consumption">Consumption:</label>
                        <input type="text" class="form-control" id="consumption" name="consumption" required>
                    </div>
                    <div class="form-group">
                        <label for="totalcylinderoilconsumption">Total Cylinder Oil Consumption:</label>
                        <input type="text" class="form-control" id="totalcylinderoilconsumption" name="totalcylinderoilconsumption" required>
                    </div>
                    <div class="form-group">
                        <label for="totalcylinderoilspecificconsumption">Total Cylinder Oil Specific Consumption:</label>
                        <input type="text" class="form-control" id="totalcylinderoilspecificconsumption" name="totalcylinderoilspecificconsumption" required>
                    </div>
                    <div class="form-group">
                        <label for="saileddistance">Sailed Distance:</label>
                        <input type="text" class="form-control" id="saileddistance" name="saileddistance" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Predict</button>
                </form>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.6.0/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
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
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fuel Consumption Prediction Result</title>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center">Fuel Consumption Prediction</h1>
                <div class="card mt-4">
                    <div class="card-body">
                        <p><strong>Air Pressure:</strong> {{ airpressure }}</p>
                        <p><strong>Consumption:</strong> {{ consumption }}</p>
                        <p><strong>Total Cylinder Oil Consumption:</strong> {{ totalcylinderoilconsumption }}</p>
                        <p><strong>Total Cylinder Oil Specific Consumption:</strong> {{ totalcylinderoilspecificconsumption }}</p>
                        <p><strong>Sailed Distance:</strong> {{ saileddistance }}</p>
                        <h2 class="mt-4"><strong>Total Consumption:</strong> {{ total_consumption }}</h2>
                        <h2><strong>Fuel Per Nautical Mile:</strong> {{ fuel_per_nautical_mile }}</h2>
                    </div>
                </div>
                <div class="text-center mt-3">
                    <a href="/" class="btn btn-primary">Back to Home</a>
                </div>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.6.0/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
    ''', airpressure=airpressure, consumption=consumption, totalcylinderoilconsumption=totalcylinderoilconsumption, totalcylinderoilspecificconsumption=totalcylinderoilspecificconsumption, saileddistance=saileddistance, total_consumption=total_consumption, fuel_per_nautical_mile=fuel_per_nautical_mile)

if __name__ == '__main__':
    try:
        load_saved_artifacts()
        print("Server is running.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        exit(1)
    except AttributeError as e:
        print(f"Error in model attributes: {e}")
        exit(1)
    app.run(debug=True)


