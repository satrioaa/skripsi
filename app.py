import pandas as pd

# Load data from an Excel file
def load_data_from_excel(file_path):
    return pd.read_excel(file_path)

# Main function
def main():
    st.title("Time Series Forecasting App")

    # Load data from Excel file
    file_path = "skripsi.xlsx"
    data = load_data_from_excel(file_path)

import numpy as np
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Fungsi untuk menghasilkan dataset time series dengan window width
def generate_time_series(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Normalisasi data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    features = np.random.rand(100, 2)
    target = np.random.rand(100)
    return features, target

# Build LSTM model
def build_lstm_model(X_train):
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    return model_lstm

# Build GRU model
def build_gru_model(X_train):
    model_gru = Sequential()
    model_gru.add(GRU(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_gru.add(Dense(1))
    model_gru.compile(optimizer='adam', loss='mean_squared_error')
    return model_gru

# Build SVR model
def build_svr_model(training_features, training_target):
    model_svr = SVR(kernel='rbf')
    model_svr.fit(training_features, training_target)
    return model_svr

# Main function
def main():
    st.title("Time Series Forecasting App")

    # Load data
    features, target = load_data()

    # Define window size
    window_size = st.slider("Window Size:", min_value=1, max_value=10, value=5)

    # Generate dataset with window width
    normalized_features = normalize_data(features)
    X, y = generate_time_series(normalized_features, window_size)

    # Reshape input data for LSTM and GRU
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, shuffle=False)

    # Build models
    model_lstm = build_lstm_model(X_train)
    model_gru = build_gru_model(X_train)
    model_svr = build_svr_model(features, target)

    # Train models
    history_lstm = model_lstm.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)
    history_gru = model_gru.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

    # Evaluate models
    lstm_acc = model_lstm.evaluate(X_test, y_test, verbose=0)
    gru_acc = model_gru.evaluate(X_test, y_test, verbose=0)
    svr_predictions = model_svr.predict(features)
    svr_mse = mean_squared_error(target, svr_predictions)

    # Display results
    st.write('LSTM Accuracy:', lstm_acc)
    st.write('GRU Accuracy:', gru_acc)
    st.write('SVR MSE:', svr_mse)

if __name__ == "__main__":
    main()
