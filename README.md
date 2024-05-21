## Movie Rating Prediction

## Author: Sandeep Kumawat

## Batch: April 2024 (A48)

## Domain: Data Science

## Aim

The primary objective of this project is to develop a model that can predict movie ratings based on provided features.

## Datasets

This project utilized the following datasets:

1. Movie_data: Contains preprocessed movie information, including MovieName, Genre, and MovieIDs.
2. Ratings_data: Contains preprocessed ratings information, including Ratings and Timestamp.
3. Users_data: Contains preprocessed user information, including Gender, Age, Occupation, and Zip-code.

## Libraries Used

The project made use of several essential libraries:

- numpy
- pandas
- matplotlib.pyplot
- seaborn
- sklearn.preprocessing.LabelEncoder
- sklearn.preprocessing.MinMaxScaler
- sklearn.model_selection.train_test_split
- sklearn.linear_model.LogisticRegression

## Data Exploration and Preprocessing

1. Loaded Movie_data, Ratings_data, and Users_data as DataFrames from separate CSV files.
2. Eliminated missing values from each DataFrame using `dropna(inplace=True)`.
3. Displayed the shape and descriptive statistics for each DataFrame using `df.shape` and `df.describe()`.
4. Converted the 'Gender' column in Users_data from categorical to numerical values using LabelEncoder.
5. Horizontally concatenated the DataFrames using `pd.concat` to create a final dataset `df_data`.
6. Dropped unnecessary columns like 'Occupation', 'Zip-code', and 'Timestamp' from the final dataset to create `df2`.
7. Removed any remaining missing values from the final dataset `df2` using `dropna()`.
8. Conducted data visualization using count plots and histograms to analyze the distribution of ratings, genders, and age.

## Model Training

1. Created the feature matrix `input` and target vector `target` using relevant columns from the final dataset `df_final`.
2. Split the data into training and testing sets using `train_test_split`.
3. Scaled the input data using MinMaxScaler to normalize the values between 0 and 1.

## Model Prediction

1. Initialized a logistic regression model and trained it on the training data using `LogisticRegression`.
2. Employed the trained model to predict movie ratings for the test set using `model.predict(X_test)`.
