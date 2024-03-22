from flask import Flask, request , render_template
import numpy as np 
import joblib

app = Flask(__name__)

@app.route('/')
def func1():
    return render_template('home.html')

@app.route('/predict', methods = ['get','post'])
def func2():
    temp = request.form.get('string')

    model = joblib.load('best_models/naive_bayes.pkl')

    pred = model.predict([temp])
    return render_template('predict.html',temp = str(pred))

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 5000)