import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entred the data ingestion method or component.")
        try:
            df=pd.read_csv('D:/AllFam/ML/Student_performance/notebook/data/student.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)  # save Raw data

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=40)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)  # save Raw data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)  # save Raw data
            
            logging.info("Injestion of data is completed")
            return(
                    self.ingestion_config.train_data_path ,
                    self.ingestion_config.test_data_path 
            )
        except Exception as e:
            raise Exception(e,sys)
        
if __name__ =='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()