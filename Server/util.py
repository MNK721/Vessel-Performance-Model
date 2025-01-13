

__data_columns = None
__model = None

def estimate_fuel_consumption(airpressure, consumption, totalcylinderoilconsumption, saileddistance):
    x = np.zeros(len(__data_columns))
    x[__data_columns.index('airpressure')] = airpressure
    x[__data_columns.index('consumption')] = consumption
    x[__data_columns.index('totalcylinderoilconsumption')] = totalcylinderoilconsumption
    x[__data_columns.index('saileddistance')] = saileddistance

 fuel_per_nautical_mile = totalcylinderoilconsumption/ saileddistance

    return fuel_per_nautical_mile




if __name__ == "__main__":
    load_saved_artifacts()
    print(estimate_fuel_consumption(1016, 70, 100, 23.80556))
    print(estimate_fuel_consumption(1015, 70, 100, 23.80556))
    app.run(debug=True)