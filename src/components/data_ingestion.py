# py for data ingestion #

# load the data and prepare for the analysis

import os
import sys

from src.exception import CustomException
# importing and using the CustomException created previously

from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:

	# path for saving the train data
	train_data_path: str=os.path.join("artifacts", "train.csv")

	test_data_path: str=os.path.join("artifacts", "test.csv")

	raw_data_path: str=os.path.join("artifacts", "data.csv")


class DataIngestion:
	def __init__(self):
		self.ingestion_config = DataIngestionConfig()

	def initiate_data_ingestion(self):
		logging.info("Entered the data ingestion method or component")
		# woah.. so this is how to log what's happening?

		try:
			df = pd.read_csv("data/stud.csv")
			# if you have to read from MongoDB or wtv,
			# you could do that instead

			logging.info("Read the dataset as dataframe")

			# making a directory
			os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

			df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

			logging.info("Train-test split initiated")
			train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 22082023)

			train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

			test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

			logging.info("Ingestion of the data completed. Raw, Train, Test sets all saved")

			return(
					self.ingestion_config.train_data_path,
					self.ingestion_config.test_data_path
					)

		except Exception as e:
			raise CustomException(e, sys)


# testing
if __name__ == "__main__":
	obj = DataIngestion()
	train_data_path,test_data_path=obj.initiate_data_ingestion()

	data_transformation = DataTransformation()
	# running the DataTransformation class

	# executing the initiate_data_transformation
	train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
	# the function returns the train and test set as np.arrays,
	# and also saves the preprocessing pipeline (that contains the)
	# model used for preprocessing the data, with pickle (dill)

	model_trainer = ModelTrainer()
	print(model_trainer.initiate_model_trainer(train_arr, test_arr))







