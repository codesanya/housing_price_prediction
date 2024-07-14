
# World Real Estate Data Analysis and Prediction

## Overview

This project focuses on analyzing and predicting housing prices using machine learning techniques. The dataset used contains approximately 147,000 entries from global real estate markets.

## Data Description

The dataset includes various features such as:

- **title**: Title of the property listing.
- **country**: Country where the property is located.
- **location**: Specific location within the country.
- **building_construction_year**: Year in which the building was constructed.
- **building_total_floors**: Total number of floors in the building.
- **apartment_total_area**: Total area of the apartment in square meters.
- **apartment_living_area**: Living area of the apartment in square meters.
- **price_in_USD**: Price of the property in US dollars.
- **image**: URL link to the property's image.
- **url**: URL link to the property's listing page.

## Workflow

### 1. Data Preprocessing

#### Handling Missing Values

- Removed rows where 'price_in_USD' is missing since it's the target variable.
- Filled missing numeric values ('apartment_total_area', 'apartment_living_area') using nearest neighbor imputation based on 'country', 'location', and 'price_in_USD'.

#### Encoding Categorical Features

- Encoded categorical features ('title', 'country', 'location') using LabelEncoder.

### 2. Feature Engineering

- Extracted additional features such as:
  - Property age from 'building_construction_year'.
  - Living area ratio from 'apartment_living_area' and 'apartment_total_area'.
  - Extracted image features using a pre-trained ResNet50V2 model for image URLs.

### 3. Modeling

#### Random Forest Regression

- Trained a Random Forest Regressor with 200 estimators after standardizing the data.
- Tuned hyperparameters using GridSearchCV to optimize performance.
- Evaluated model performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

#### XGBoost Regression

- Utilized XGBoost Regressor for comparison, optimizing hyperparameters via GridSearchCV.
- Assessed model accuracy using MAE, MSE, and R² Score metrics.

### 4. Results

- **Random Forest**:
  - Mean Absolute Error: 22833.60
  - Mean Squared Error: 2070649614.68
  - R² Score: 0.841

- **XGBoost**:
  - Mean Absolute Error: 23877.50
  - Mean Squared Error: 1741630575.80
  - R² Score: 0.866

## Prerequisites

- Python 3
- Pandas
- Scikit-learn
- XGBoost
- TensorFlow
- Requests
- PIL
- tqdm


## Authors

Sanya Vishwakarma

## License

This project is licensed under the MIT License 
