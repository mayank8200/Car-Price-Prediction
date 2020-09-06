import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('regressor.pkl', 'rb'))
lb = pickle.load(open('lb', 'rb'))
lb1 = pickle.load(open('lb1', 'rb'))
lb2 = pickle.load(open('lb2', 'rb'))
lb3 = pickle.load(open('lb3', 'rb'))
lb0 = pickle.load(open('lb0', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    year = int(features[1])
    kms_driven = int(features[2])
    final_features = []
    #a=[]
    #a = 
    #print(a)
    final_features.append(lb0.transform([features[0]])[0])
    final_features.append(year)
    final_features.append(kms_driven)
    final_features.append(lb.transform([features[3]])[0])
    final_features.append(lb1.transform([features[4]])[0])
    final_features.append(lb2.transform([features[5]])[0])
    final_features.append(lb3.transform([features[6]])[0])
    print(len(final_features))
    prediction = model.predict([final_features])
    output = round(prediction[0], 2)    
    return render_template('index.html',prediction_text = output)
 


if __name__ == "__main__":
    app.run(debug=True)