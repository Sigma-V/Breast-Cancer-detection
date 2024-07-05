from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']
        
        # Read the file and extract parameters (assuming CSV format)
        data = np.loadtxt(file, delimiter=',')
        
        # Make predictions
        predictions = model.predict(data.reshape(1, -1))
        
        # Display the result
        result = 'Positive for breast cancer' if predictions[0] == 1 else 'Negative for breast cancer'
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
