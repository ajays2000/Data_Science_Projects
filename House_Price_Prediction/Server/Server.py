from flask import Flask, request, jsonify
from flask_cors import CORS

import util
app = Flask(__name__)
CORS(app)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
                        'locations' : util.get_location_names()
                       })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/get_estimated_price', methods = ['POST'])
def get_estimated_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
                        'estimated_price' : util.get_estimated_price(location, total_sqft, bhk, bath)
                       })
    return response

if __name__ == "__main__":
    print("Starting Python Flask Sever For Home Price Prediction.....")
    util.load_saved_artifacts()
    app.run()
