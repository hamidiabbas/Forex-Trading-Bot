"""
/******************************************************************************
 *
 * FILE NAME:           train_model.py (with Evaluation - Corrected)
 *
 * PURPOSE:
 *
 * This version corrects an IndentationError and includes the full,
 * functional code for generating data and training the model.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             47.1 (with Evaluation - Corrected)
 *
 ******************************************************************************/
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt

import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence

# --- 1. CONFIGURATION ---
SYMBOL = 'EURUSD'
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
SEQUENCE_LENGTH = 20

def create_training_data():
    """ Fetches data and engineers features for model training. """
    print("--- Stage 1: Creating Training Data ---")
    
    data_handler = DataHandler(config)
    market_intel = MarketIntelligence(data_handler, config)
    
    data_handler.connect()
    from datetime import datetime
    import pytz
    timezone = pytz.timezone("Etc/UTC")
    start_date_dt = datetime.strptime(START_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    end_date_dt = datetime.strptime(END_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    
    df = data_handler.get_data_by_range(SYMBOL, TIMEFRAME, start_date_dt, end_date_dt)
    data_handler.disconnect()
    
    if df is None or df.empty:
        print("Could not fetch data. Aborting.")
        return None

    print("Calculating features...")
    df_features = market_intel._analyze_data(df.copy())
    
    df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    
    df_features.dropna(inplace=True)
    
    print(f"Feature engineering complete. Dataset has {len(df_features)} bars.")
    df_features.to_csv(f"training_data_{SYMBOL}.csv")
    print(f"Saved training data to 'training_data_{SYMBOL}.csv'")
    return df_features

def evaluate_model(model, X_test, y_test):
    """ Evaluates the trained model and prints a performance report. """
    print("\n--- Stage 3: Evaluating Model Performance ---")
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Check if a graphical environment is available before showing plot
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
    except Exception as e:
        print(f"\nCould not display plot. Your environment may not support graphics. Error: {e}")


def train_model(df):
    """ Builds, trains, evaluates, and saves the LSTM model. """
    print("\n--- Stage 2: Building and Training Model ---")
    
    # --- THIS IS THE FIX ---
    # The columns 'returns' and 'autocorr' were removed as they are no longer calculated.
    features = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'hurst'])
    target = df['target'].values
    
    print("Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        X.append(scaled_features[i-SEQUENCE_LENGTH:i])
        y.append(target[i])
    X, y = np.array(X), np.array(y)
    
    # Split data without shuffling to preserve time-series order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print(f"Data split: {len(X_train)} training sequences, {len(X_test)} testing sequences.")

    print("Building LSTM model...")
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(256, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    model.save(f"model_{SYMBOL}.h5")
    print(f"\nTraining complete. Model saved to 'model_{SYMBOL}.h5'")

    evaluate_model(model, X_test, y_test)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    model.save(f"model_{SYMBOL}.h5")
    print(f"\nTraining complete. Model saved to 'model_{SYMBOL}.h5'")

    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    training_df = create_training_data()
    if training_df is not None:
        train_model(training_df)