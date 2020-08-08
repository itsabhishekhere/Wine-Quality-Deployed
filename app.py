from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Model
filename = 'rf_classifier_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        wine_type = int(request.form['type'])
        fixed_acidity = float(request.form['fixed acidity'])
        volatile_acidity = float(request.form['volatile acidity'])
        citric_acid = float(request.form['citric acid'])
        residual_sugar = float(request.form['residual sugar'])
        chlorides = float(request.form['chlorides'])
        free_sd = float(request.form['free sulfur dioxide'])
        total_sd = float(request.form['total sulfur dioxide'])
        density = float(request.form['density'])
        ph = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        input_data = np.array([[wine_type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sd, total_sd, density, ph, sulphates, alcohol]])

        my_prediction = classifier.predict(input_data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)