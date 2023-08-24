# model_trainer #
# py file for training the model


import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd


from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
	AdaBoostRegressor,
	GradientBoostingRegressor,
	RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score


# exception handling, logging and utils
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# for every module, we need a config file
@dataclass
class ModelTrainerConfig:
	trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
	def __init__(self):
		self.model_trainer_config = ModelTrainerConfig()
		# getting the path name using the class above

	def initiate_model_trainer(self, train_array, test_array):

		try:
			logging.info("Reading in the test and test array")

			X_train,y_train,X_test,y_test = (
				train_array[:, :-1],
				train_array[:, -1],
				test_array[:, :-1],
				test_array[:, -1]
			)


			# print("X_train \n", X_train[:2,])

			# print("X_test: \n", X_test[:2,])

			# defining the dictionary of models:
			models = {
			"Random Forest": RandomForestRegressor(),
			"Decision Tree": DecisionTreeRegressor(),
			"Gradient Boosting": GradientBoostingRegressor(),
			"Linear Regression": LinearRegression(),
			"XGBRegressor": XGBRegressor(),
			"CatBoosting Regressor": CatBoostRegressor(verbose=False),
			"AdaBoost Regressor": AdaBoostRegressor(),
			}

			# specifying the hyperparameters for model selection:
			params={
				"Decision Tree": {
					'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
					# 'splitter':['best','random'],
					# 'max_features':['sqrt','log2'],
					},

				"Random Forest":{
					# 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
					# 'max_features':['sqrt','log2',None],
					'n_estimators': [8,32,256]
					},
				"Gradient Boosting":{
					# 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
					'learning_rate':[.1,.01,.001],
					'subsample':[0.6,0.9],
					# 'criterion':['squared_error', 'friedman_mse'],
					# 'max_features':['auto','sqrt','log2'],
					'n_estimators': [8,256]
					},

				"Linear Regression":{},

				"XGBRegressor":{
					'learning_rate':[.1,.01,.001],
					'n_estimators': [8,256]
					},

				"CatBoosting Regressor":{
					'depth': [6,10],
					'learning_rate': [0.01, 0.05, 0.1],
					'iterations': [30, 100]
					},
				"AdaBoost Regressor":{
					'learning_rate':[.1,.01,.001],
					# 'loss':['linear','square','exponential'],
					'n_estimators': [8,256]
					}
				}

			logging.info("Initiating model fitting.")

			# the evaluate_model function is defined on the
			# utils module
			model_report:dict=evaluate_models(X_train=X_train, 
				y_train=y_train, 
				X_test=X_test, 
				y_test=y_test, 
				models=models,
				param=params
			)

			# best model score from the dictionary
			best_model_score = max(sorted(model_report.values()))

			best_model_name = list(model_report.keys())[
				list(model_report.values()).index(best_model_score)
			]

			best_model = models[best_model_name]

			if best_model_score < 0.6:
				raise CustomException("All the models are below 60% R2.")

			logging.info("R2 evaluated on the basis of test data.")

			# saving the best model as the pkl file
			save_object(
				file_path = self.model_trainer_config.trained_model_file_path,
				obj = best_model
			)

			predicted = best_model.predict(X_test)
			predicted_r2_square =  r2_score(y_test, predicted)

			return best_model, predicted_r2_square

		except Exception as e:
			raise CustomException(e, sys)
