from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            fixed_acidity= float(request.form.get('fixed_acidity')),
            volatile_acidity= float(request.form.get('volatile_acidity')),
            citric_acid= float(request.form.get('citric_acid')),
            residual_sugar= float(request.form.get('residual_sugar')),
            chlorides= float(request.form.get('chlorides')),
            free_sulfur_dioxide= float(request.form.get('free_sulfur_dioxide')),
            total_sulfur_dioxide= float(request.form.get('total_sulfur_dioxide')),
            density= float(request.form.get('density')),
            pH= float(request.form.get('pH')),
            sulphates= float(request.form.get('sulphates')),
            alcohol= float(request.form.get('alcohol'))
        )

        pred_df = data.get_data_as_data_frame()

        logging.info("DataFrame for Prediction:")
        logging.info(str(data))
        logging.info("Before Prediction")

        predict_pipeline = PredictPipeline()
        logging.info("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        logging.info("After Prediction")

        return render_template('home.html', results= results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001)        


