# Import all the required libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformantion, DataTransformantionConfig
from src.components.model_train import ModelTrainer,ModelTrainerConfig
# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Data ingestion method Started")
        try:
            df=pd.read_csv('D:/AllFam/ML/Student_performance/notebook/data/student.csv')
            logging.info('Read the dataset as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)  # save Raw data

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=40)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)  # save train data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)  # save test data
            
            logging.info("Injestion of data is completed")

            return(
                    self.ingestion_config.train_data_path ,
                    self.ingestion_config.test_data_path 
            )
        

        except Exception as e:
            logging.info("Exception occured at data Injestion ")
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformantion()
    train_arr, test_arr,_ =data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



