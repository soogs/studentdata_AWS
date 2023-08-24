# prediction pipeline # 
# this is used at the deployment phase #

import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
	def __init__(self):
		pass

	def predict(self, features_df):
		# performing the prediction here

		try:
			# getting the pickles 
			model_path= "artifacts/model.pkl"
			preprocessor_path = "artifacts/preprocessor.pkl"

			# loading the pickle object (model)
			model = load_object(file_path = model_path)
			preprocessor = load_object(file_path = preprocessor_path)

			# this takes in the new data, and runs it to the pipeline! cool
			data_processed = preprocessor.transform(features_df)
			preds = model.predict(data_processed)

			return preds

		except Exception as e:
			raise CustomException(e, sys)



class CustomData:
	# this is responsible in mapping the input of the HTML 
	# to bringing it here to the backend

	def __init__(self,
		gender: str,
		race_ethnicity:str,
		parental_level_of_education:str,
		lunch:str,
		test_preparation_course:str,
		reading_score:int,
		writing_score:int):

		# assigining the values
		self.gender = gender
		self.race_ethnicity = race_ethnicity
		self.parental_level_of_education = parental_level_of_education
		self.lunch = lunch
		self.test_preparation_course = test_preparation_course
		self.reading_score = reading_score
		self.writing_score = writing_score

	def get_data_as_data_frame(self):
		# converting the input data into a dataframe

		try:
			custom_data_input_dict = {
				"gender": [self.gender],
				"race_ethnicity": [self.race_ethnicity],
				"parental_level_of_education": [self.parental_level_of_education],
				"lunch": [self.lunch],
				"test_preparation_course": [self.test_preparation_course],
				"reading_score": [self.reading_score],
				"writing_score": [self.writing_score]
			}

			return pd.DataFrame(custom_data_input_dict)

		except Exception as e:
			raise CustomException(e, sys)

