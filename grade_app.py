import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load your pre-trained model (if available)
# If you don't have a model saved yet, you can skip this section and train a model directly inside Streamlit.
# model = pickle.load(open('linear_model.pkl', 'rb'))

# Title of the web app
st.title('Student CGPA Prediction')

# File uploader for dataset
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Display Boxplots for feature outlier detection
    st.subheader('Boxplot for Feature Outlier Detection')
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df)
    st.pyplot()

    # Calculate IQR for Outlier Removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    st.write("Data After Removing Outliers:", df_no_outliers.head())

    # Feature selection: Assuming the target variable is 'CGPA' and others are features
    X = df_no_outliers.drop(columns=['CGPA'])  # Replace with your actual target column
    y = df_no_outliers['CGPA']

    # Train a simple Linear Regression model
    st.subheader('Training a Linear Regression Model')

    if st.button('Train Model'):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')

        # Save the model using pickle
        with open('linear_model.pkl', 'wb') as file:
            pickle.dump(model, file)

        st.write("Model saved successfully!")

    # Make Predictions Using the Model
    st.subheader('Predict CGPA for New Data')

    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f"Enter {column}", value=0.0)

    if st.button('Predict CGPA'):
        input_df = pd.DataFrame(input_data, index=[0])
        prediction = model.predict(input_df)
        st.write(f"Predicted CGPA: {prediction[0]}")

    # Additional Option to download the predictions
    st.subheader('Download Predictions')
    if st.button('Download Predictions'):
        predictions = model.predict(X)
        df_no_outliers['Predicted_CGPA'] = predictions
        df_no_outliers.to_csv('predictions.csv', index=False)
        st.write("Predictions saved as 'predictions.csv'")
