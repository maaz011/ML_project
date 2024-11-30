import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  #for handling missing  values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from src.exeception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass           #it provide inputs to the data transformtion conponent
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

# encoder = OneHotEncoder(handle_unknown='ignore')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):   #    create all pickle files respnsible for converting cateogrical f to num. f//data transformation
        try:
            numerical_columns=["year","km_driven","seats"]
            categorical_columns=["fuel",
                                 "seller_type",
                                 "transmission",
                                 "owner",
                                 "mileage",
                                 "engine",
                                 "max_power",
                                 "brands",
                                 "car_names",
                                 "variants",
                                 "insurance"

            ]    #name,year,selling_price,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("target_encoder", ce.TargetEncoder()),   #handle_unknown='ignore'
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read tain and test data completed")
            logging.info("obtaining preprocessing obj")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="selling_price"
            numerical_column=["year","km_driven","seats"]  #year,selling_price,km_driven,seats
            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"applying preprocessing obj on training dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")

            target_feature_train_arr = np.array(target_feature_train_df)      #.reshape(-1, 1)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]         #.reshape(-1, 1)
            
            logging.info("saved preprocessing obj")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            
        except Exception as e:
            raise CustomException(e,sys)


# code 2:
# import sys
# from dataclasses import dataclass

# import numpy as np 
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler

# from src.exeception import CustomException
# from src.logger import logging
# import os

# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config=DataTransformationConfig()

#     def get_data_transformer_object(self):
#         '''
#         This function si responsible for data trnasformation
        
#         '''
#         try:
#             numerical_columns = ["year","km_driven","seats"]
#             categorical_columns = [
#               "name",
#               "fuel",
#               "seller_type",
#               "transmission",
#               "owner",
#               "mileage",
#               "engine",
#               "max_power",
#               "torque"
#             ]

#             num_pipeline= Pipeline(
#                 steps=[
#                 ("imputer",SimpleImputer(strategy="median")),
#                 ("scaler",StandardScaler())

#                 ]
#             )

#             cat_pipeline=Pipeline(

#                 steps=[
#                 ("imputer",SimpleImputer(strategy="most_frequent")),
#                 ("one_hot_encoder",OneHotEncoder()),
#                 ("scaler",StandardScaler(with_mean=False))
#                 ]

#             )

#             logging.info(f"Categorical columns: {categorical_columns}")
#             logging.info(f"Numerical columns: {numerical_columns}")

#             preprocessor=ColumnTransformer(
#                 [
#                 ("num_pipeline",num_pipeline,numerical_columns),
#                 ("cat_pipelines",cat_pipeline,categorical_columns)

#                 ]


#             )

#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def initiate_data_transformation(self,train_path,test_path):

#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)

#             logging.info("Read train and test data completed")

#             logging.info("Obtaining preprocessing object")

#             preprocessing_obj=self.get_data_transformer_object()

#             target_column_name="selling_price"
#             numerical_columns = ["year","km_driven","seats"]

#             input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
#             target_feature_train_df=train_df[target_column_name]

#             input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
#             target_feature_test_df=test_df[target_column_name]

#             logging.info(
#                 f"Applying preprocessing object on training dataframe and testing dataframe."
#             )

#             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


#             train_arr = np.c_[
#                 input_feature_train_arr, np.array(target_feature_train_df)
#             ]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info(f"Saved preprocessing object.")

#             save_object(

#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj

#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e,sys)
