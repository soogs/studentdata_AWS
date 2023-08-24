# application.py #
# module for creating the Flask App
# for AWS deployment, "application" is the convention used

import pickle
import dill
from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# importing the CustomData class that takes the input in and makes it into df
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
	return render_template("index.html")


# specifying two methods of GET and POST
@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
	# getting the data, and doing the prediction here
	if request.method == 'GET':
		return render_template('home.html')
		# for the method GET, we just have the input field for the users

	else:
		# let's create the data for the POST method

		# 'request' collects the input data on the web app
		data = CustomData(
			gender = request.form.get('gender'),
			race_ethnicity = request.form.get('race_ethnicity')	,
			parental_level_of_education = request.form.get('parental_level_of_education'),
			lunch = request.form.get('lunch'),
			test_preparation_course = request.form.get('test_preparation_course'),
			reading_score = float(request.form.get('reading_score')),
			writing_score = float(request.form.get('writing_score'))
		)

		# this 'get_data_as_data_frame' function is within the CustomData class
		pred_df = data.get_data_as_data_frame()

		print(pred_df)

		# initialize the pipeline for prediction
		predict_pipeline = PredictPipeline()
		results = predict_pipeline.predict(pred_df)

		return render_template("home.html", results = results[0])

if __name__ == "__main__":
	app.run(host = "0.0.0.0")
	# appraently for deployment, this debug = True should be removed


		

