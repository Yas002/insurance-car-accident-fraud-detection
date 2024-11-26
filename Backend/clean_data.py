# this file is a pipeline for preprocessing the data in production before it is passed to the model for prediction
# note: the data should be csv format
"""
the input dataframe to this pipeline should have this schema:
    Month : object
    Make : object
    Sex : object
    MaritalStatus : object
    Age : int64
    Fault : object
    VehicleCategory : object
    VehiclePrice : object
    RepNumber : int64
    Deductible : int64
    DriverRating : int64
    PastNumberOfClaims : object
    AgeOfVehicle : object
    WitnessPresent : object
    AgentType : object
    NumberOfSuppliments : object
    AddressChange_Claim : object
    NumberOfCars : object
    Year : int64
    BasePolicy : object


the output dataframe will have this schema:
    Month: int
    Sex: int
    MaritalStatus: int
    Age: int
    Fault: int
    VehicleCategory: int
    VehiclePrice: int
    RepNumber: int
    Deductible: int
    DriverRating: int
    PastNumberOfClaims: int
    AgeOfVehicle: int
    NumberOfSuppliments: int
    AddressChange_Claim: int
    NumberOfCars: int
    Year: int
    BasePolicy: int
    Make_Honda: int
    Make_Mazda: int
    Make_Pontiac: int
    Make_Toyota: int
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



def encode_binary_columns(df, binary_features):
    # encoding binaray categorigal variables
    le = LabelEncoder()
    for col  in binary_features:
        df[col] = le.fit_transform(df[col])
    return df




def map_categorical_columns(df):
    # Map categorical columns to numeric values in the DataFrame.
    # Define mappings
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    vehicleprice_mapping = {
        'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2,
        '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5
    }

    ageofvehicle_mapping = {
        'new': 2, '2 years': 0, '3 years': 2, '4 years': 2,
        '5 years': 1, '6 years': 1, '7 years': 0, 'more than 7': 0
    }

    basepolicy_mapping = {'Liability': 0, 'Collision': 1, 'All Perils': 2}
    materialstatus_mapping = {'Single': 0, 'Married': 1, 'Widow': 2, 'Divorced': 3}

    vehicle_category_mapping = {
        'Sport': 0,
        'Sedan': 1,
        'Utility': 2
    }

    past_number_of_claims_mapping = {
        'none': 0,
        '1': 1,
        '2 to 4': 2,
        'more than 4': 3
    }

    number_of_suppliments_mapping = {
        'none': 0,
        '1 to 2': 1,
        '3 to 5': 2,
        'more than 5': 3
    }

    address_change_claim_mapping = {
        'no change': 0,
        'under 6 months': 1,
        '1 year': 2,
        '2 to 3 years': 3,
        '4 to 8 years': 4
    }

    number_of_cars_mapping = {
        '1 vehicle': 0,
        '2 vehicles': 1,
        '3 to 4': 2,
        '5 to 8': 3,
        'more than 8': 4
    }

    # Apply mappings
    df['Month'] = df['Month'].map(month_mapping)
    df['VehiclePrice'] = df['VehiclePrice'].map(vehicleprice_mapping)
    df['AgeOfVehicle'] = df['AgeOfVehicle'].map(ageofvehicle_mapping)
    df['BasePolicy'] = df['BasePolicy'].map(basepolicy_mapping)
    df['MaritalStatus'] = df['MaritalStatus'].map(materialstatus_mapping)
    df['PastNumberOfClaims'] = df['PastNumberOfClaims'].map(past_number_of_claims_mapping)
    df['NumberOfSuppliments'] = df['NumberOfSuppliments'].map(number_of_suppliments_mapping)
    df['AddressChange_Claim'] = df['AddressChange_Claim'].map(address_change_claim_mapping)
    df['NumberOfCars'] = df['NumberOfCars'].map(number_of_cars_mapping)
    df['VehicleCategory'] = df['VehicleCategory'].map(vehicle_category_mapping)

    return df




def add_make_columns(df):
    # Check if 'Make' column exists in the DataFrame
    if 'Make' in df.columns:
        # Check for each Make and add the corresponding column with value 0
        if (df['Make'] == 'Honda').any():
            df['Make_Honda'] = 1
            df['Make_Mazda'] = 0
            df['Make_Pontiac'] = 0
            df['Make_Toyota'] = 0
        if (df['Make'] == 'Mazda').any():
            df['Make_Honda'] = 0
            df['Make_Mazda'] = 1
            df['Make_Pontiac'] = 0
            df['Make_Toyota'] = 0
        if (df['Make'] == 'Pontiac').any():
            df['Make_Honda'] = 0
            df['Make_Mazda'] = 0
            df['Make_Pontiac'] = 1
            df['Make_Toyota'] = 0
        if (df['Make'] == 'Toyota').any():
            df['Make_Honda'] = 0
            df['Make_Mazda'] = 0
            df['Make_Pontiac'] = 0
            df['Make_Toyota'] = 1
    df.drop(columns=['Make'], inplace=True)
    return df


def replace_age_zero(df):
    # Replacing the Age = 0 with the median 
    age_median = 38
    df['Age'].replace(0, age_median)
    return df



def remove_constant_variance_features(df):
    constant_variance_features = ['WitnessPresent', 'AgentType', 'Make_BMW', 'Make_Dodge', 'Make_Mercury',
       'Make_Nisson', 'Make_Saab', 'Make_Saturn', 'Make_VW']
    
    # Get features that exist in DataFrame
    features_to_drop = [col for col in constant_variance_features if col in df.columns]
    
    if features_to_drop:
        return df.drop(columns=features_to_drop)
    return df






def create_preprocessing_pipeline(df):
    print("Starting preprocessing pipeline for production data...")
    print("Applying preprocessing pipeline...")
    # get the categorical binary features and encode it
    binary_features = ['Sex', 'Fault', 'WitnessPresent', 'AgentType']
    df = encode_binary_columns(df, binary_features)

    # maping those categorical features  ['Month', 'VehiclePrice', 'AgeOfVehicle', 'BasePolicy', 'MaritalStatus',
    # 'PastNumberOfClaims', 'NumberOfSuppliments', 'AddressChange_Claim',
    # 'NumberOfCars', 'PolicyType', 'VehicleCategory'] and change it into numerical
    df = map_categorical_columns(df)

    # add make columns if it does not exist
    df = add_make_columns(df)

    # replacing the age value 0 with the median
    df = replace_age_zero(df)

    # remove the values that has variance < 0.1 using the VarianceThreshold method
    df = remove_constant_variance_features(df)

    print("✓ Preprocessing completed successfully!")

    return df