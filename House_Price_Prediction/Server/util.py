import warnings
import pickle
import numpy as np
warnings.filterwarnings('ignore')

import json
__locations =  None
__data_columns =  None
__model = None


def get_estimated_price(location, sqft, bhk, bath):

    try:

        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x1 = np.zeros(len(__data_columns))
    x1[0] = sqft
    x1[1] = bath
    x1[2] = bhk

    if loc_index >= 0:
        x1[loc_index] = 1

    return round(__model.predict([x1])[0])
def get_location_names():
    return __locations

def load_saved_artifacts():
    print('loading saved artifacts....start')
    global __data_columns
    global __locations

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open('./artifacts/bangalore_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)
    print('loading saved artifacts.....done')


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(float(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3)))
    print(float(get_estimated_price('Kalhalli', 1000, 2, 2,))) #other location
    print(float(get_estimated_price('Ejipura', 1000, 2, 2))) #other location