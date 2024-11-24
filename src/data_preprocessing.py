# this file creates a pipeline for preprocessing the data and saving the preprocessed datasets to files as train and test for both folds (the firstfold to train the models on
# and the second fold to retrain the model considering the new data)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold



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

    days_policy_claim_mapping = {
        'none': 0,
        '8 to 15': 1,
        '15 to 30': 2,
        'more than 30': 3
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
    df['Days_Policy_Claim'] = df['Days_Policy_Claim'].map(days_policy_claim_mapping)
    df['PastNumberOfClaims'] = df['PastNumberOfClaims'].map(past_number_of_claims_mapping)
    df['NumberOfSuppliments'] = df['NumberOfSuppliments'].map(number_of_suppliments_mapping)
    df['AddressChange_Claim'] = df['AddressChange_Claim'].map(address_change_claim_mapping)
    df['NumberOfCars'] = df['NumberOfCars'].map(number_of_cars_mapping)
    df['VehicleCategory'] = df['VehicleCategory'].map(vehicle_category_mapping)

    return df




def one_hot_encoding_function(df):
    df["Make"] = df["Make"].astype(str)
    df = pd.get_dummies(df, columns=["Make"])
    return df



def removing_onehot_constant_features(df):
    onehot_encoded_columns = [col for col in df.columns if '_' in col]
    constant_features = []
    for col in onehot_encoded_columns:
        if df[col].sum() <= 6:
            constant_features.append(col)
    df.drop(columns=constant_features, axis=1)
    return df


def replace_age_zero(df):
    # Replacing the Age = 0 with the median 
    age_median = df['Age'].median().astype(int)
    df['Age'].replace(0, age_median)
    return df



def remove_constant_variance_features(df, threshold):
    selector = VarianceThreshold(threshold=threshold) # The larger the threshold, the more features are eliminated.
    target_df = df['FraudFound_P']
    df_without_target = df.drop(columns = "FraudFound_P")

    reduced_array = selector.fit_transform(df_without_target)
    reduced_df = pd.DataFrame(reduced_array, columns=df_without_target.columns[selector.get_support()])
    return pd.concat([reduced_df, target_df], axis = 1)



def train_test_split_func(df, test_size_param, random_state, target_column = 'FraudFound_P'):
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=random_state)
    return X_train, X_test, y_train, y_test



def smote_oversampling(X_train, y_train, random_state=0):
    # applies SMOTE oversampling technique on train data with a given random state

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Display information before and after applying SMOTE
    print("Taining data before SMOTE: ", X_train.shape, y_train.shape)
    print("Training data after SMOTE: ", X_train_smote.shape, y_train_smote.shape)
    print()
    print("After SMOTE Label Distribution: ", pd.Series(y_train_smote).value_counts())

    # Create a DataFrame of the SMOTE-resampled training data
    df_smote_train = pd.concat([X_train_smote, y_train_smote], axis=1)
    print("Resampled DataFrame shape: ", df_smote_train.shape)

    return df_smote_train


def stratified_split_2_folds(df, target_column = "FraudFound_P"):
    #Splits the data into 2 parts using stratified splitting to maintain the same class distribution.

        # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Initialize StratifiedKFold with 2 splits
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    
    # Get the indices for the split
    for idx1, idx2 in skf.split(X, y):
        df1 = df.iloc[idx1].reset_index(drop=True)
        df2 = df.iloc[idx2].reset_index(drop=True)
    return df1, df2







def create_preprocessing_pipeline(df):
    # start by dropping the unecessary columns (decision taken from the EDA)
    columns_to_drop = ['MonthClaimed', 'WeekOfMonth', 'DayOfWeek', 'DayOfWeekClaimed', 'WeekOfMonthClaimed', 'PolicyNumber', 'AgeOfPolicyHolder', 'PolicyType']
    df = df.drop(columns=columns_to_drop, axis=1)

    # get the categorical binary features and encode it
    binary_features = [column for column in df.columns if df[column].nunique() == 2 and df[column].dtype == 'object']
    df = encode_binary_columns(df, binary_features)

    # maping those categorical features  ['Month', 'VehiclePrice', 'AgeOfVehicle', 'BasePolicy', 'MaritalStatus',
    # 'Days_Policy_Claim', 'PastNumberOfClaims', 'NumberOfSuppliments', 'AddressChange_Claim',
    # 'NumberOfCars', 'PolicyType', 'VehicleCategory'] and change it into numerical
    df = map_categorical_columns(df)

    # One hot encoding on the feature Make
    df = one_hot_encoding_function(df, 'Make')

    # removong constant features that were one hot encoded
    df = removing_onehot_constant_features(df)

    # replacing the age value 0 with the median
    df = replace_age_zero(df)

    # remove the values that has variance < 0.1 using the VarianceThreshold method
    df = remove_constant_variance_features(df, 0.1)

    # spliting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split_func(df, 0.2, 0)

    # construction of the test dataset
    df_test = pd.concat([X_test, y_test], axis=1)

    # oversampling the train data using SMOTE method
    df_train = smote_oversampling(X_train, y_train)
    #df_train = pd.concat([X_train, y_train], axis=1)

    return df_train, df_test



def main():
    print("Starting preprocessing pipeline...")
    
    # Load data
    print("Loading data from fraud_oracle.csv...")
    df = pd.read_csv("data/fraud_oracle.csv")
    
    # Preprocessing
    print("Applying preprocessing pipeline...")
    df_train, df_test = create_preprocessing_pipeline(df)
    print("✓ Preprocessing completed successfully!")
    
    # Split train data
    print("Splitting training data into two folds...")
    df_train_1, df_train_2 = stratified_split_2_folds(df_train)
    print("✓ Training data split completed!")
    
    # Split test data
    print("Splitting test data into two folds...")
    df_test_1, df_test_2 = stratified_split_2_folds(df_test)
    print("✓ Test data split completed!")
    
    # Save datasets
    print("\nSaving datasets to files...")
    df_train_1.to_csv("data/df_train_1.csv", index=False)
    df_test_1.to_csv("data/df_test_1.csv", index=False)
    df_train_2.to_csv("data/df_train_2.csv", index=False)
    df_test_2.to_csv("data/df_test_2.csv", index=False)
    print("✓ All datasets saved successfully!")
    
    print("\n=== Preprocessing pipeline completed successfully! ===")
    print(f"Final datasets shapes:")
    print(f"Train 1: {df_train_1.shape}")
    print(f"Train 2: {df_train_2.shape}")
    print(f"Test 1: {df_test_1.shape}")
    print(f"Test 2: {df_test_2.shape}")

if __name__ == "__main__":
    main()