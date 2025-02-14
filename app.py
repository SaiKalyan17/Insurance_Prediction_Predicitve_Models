from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user inputs
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Prepare input data
        input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        premium = round(prediction[0], 2)

        return render_template('index.html', prediction=f'Predicted Insurance Premium: ${premium}')

    return render_template('index.html', prediction='')

if __name__ == '__main__':
    app.run(debug=True)
