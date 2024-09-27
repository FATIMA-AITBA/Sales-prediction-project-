# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            tv = float(request.form['TV'])
            radio = float(request.form['Radio'])
            newspaper = float(request.form['Newspaper'])

            input_features = np.array([[tv, radio, newspaper]])
            
            prediction = model.predict(input_features)
            output = round(prediction[0], 2)

            return render_template('result.html', prediction_text=f'Predicted Sales: ${output}')
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
