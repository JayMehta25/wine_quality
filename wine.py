import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import os
import urllib.request

# Path where the local CSV file will be saved
local_csv_path = "winequality-red.csv"

@st.cache_data
def load_data():
    # Try loading the data from the URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Check if the file exists locally
    if os.path.exists(local_csv_path):
        # Load from local file if available
        data = pd.read_csv(local_csv_path, sep=";")
    else:
        try:
            # Download the CSV from the URL and save it locally
            urllib.request.urlretrieve(url, local_csv_path)
            data = pd.read_csv(local_csv_path, sep=";")
        except Exception as e:
            st.error(f"Failed to load data from URL: {e}")
            return None
    
    return data

# Function to train a model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# App title
st.title("Wine Quality Classification")

# Load data
data = load_data()

if data is not None:
    st.write("## Dataset Overview")
    st.write(data.head())

    # Preprocess data
    st.write("## Data Preprocessing")
    # Create binary classification target based on quality
    data['quality_label'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)
    X = data.drop(['quality', 'quality_label'], axis=1)
    y = data['quality_label']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features (important for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = train_model(X_train_scaled, y_train)

    # Display a message when the model is trained
    st.write("### Model Trained Successfully!")

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Show performance metrics
    st.write("## Model Performance")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.4f}")

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Allow the user to test the model by inputting feature values
    st.sidebar.header("Test the Model")

    # Create sliders for each feature based on the dataset ranges
    fixed_acidity = st.sidebar.slider("Fixed Acidity", float(X['fixed acidity'].min()), float(X['fixed acidity'].max()), float(X['fixed acidity'].mean()))
    volatile_acidity = st.sidebar.slider("Volatile Acidity", float(X['volatile acidity'].min()), float(X['volatile acidity'].max()), float(X['volatile acidity'].mean()))
    citric_acid = st.sidebar.slider("Citric Acid", float(X['citric acid'].min()), float(X['citric acid'].max()), float(X['citric acid'].mean()))
    residual_sugar = st.sidebar.slider("Residual Sugar", float(X['residual sugar'].min()), float(X['residual sugar'].max()), float(X['residual sugar'].mean()))
    chlorides = st.sidebar.slider("Chlorides", float(X['chlorides'].min()), float(X['chlorides'].max()), float(X['chlorides'].mean()))
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", float(X['free sulfur dioxide'].min()), float(X['free sulfur dioxide'].max()), float(X['free sulfur dioxide'].mean()))
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", float(X['total sulfur dioxide'].min()), float(X['total sulfur dioxide'].max()), float(X['total sulfur dioxide'].mean()))
    density = st.sidebar.slider("Density", float(X['density'].min()), float(X['density'].max()), float(X['density'].mean()))
    pH = st.sidebar.slider("pH", float(X['pH'].min()), float(X['pH'].max()), float(X['pH'].mean()))
    sulphates = st.sidebar.slider("Sulphates", float(X['sulphates'].min()), float(X['sulphates'].max()), float(X['sulphates'].mean()))
    alcohol = st.sidebar.slider("Alcohol", float(X['alcohol'].min()), float(X['alcohol'].max()), float(X['alcohol'].mean()))

    # Create input array from the user input
    user_input = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    # Scale the input data
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction for the user input
    prediction = model.predict(user_input_scaled)

    # Display prediction result in an alert message at the top of the app
    if prediction[0] == 1:
        st.success("This wine is predicted to be of **Good Quality**!")
    else:
        st.error("This wine is predicted to be of **Bad Quality**.")
else:
    st.error("Data could not be loaded.")
