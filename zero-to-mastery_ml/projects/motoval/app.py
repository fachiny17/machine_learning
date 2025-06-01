from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

# Load the model ---------------------------------------------------------------------
model = load('models/car_price_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        make = request.form['make']
        colour = request.form['colour']
        odometer = float(request.form['odometer'])
        doors = float(request.form['doors'])

        # Create DataFrame
        input_data = pd.DataFrame([[make, colour, odometer, doors]], columns=[
                                  "Make", "Colour", "Odometer (KM)", "Doors"])

        # Make prediction
        try:
            predicted_price = model.predict(input_data)[0]
            prediction = f"Predicted Car Price: €{float(predicted_price):,.2f}"
        except Exception as e:
            prediction = f"Error in prediction: {str(e)}"

        return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

