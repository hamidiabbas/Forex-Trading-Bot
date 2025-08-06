"""
/******************************************************************************
 *
 * FILE NAME:           train_model.py (Ultimate V2 - All Features)
 *
 * PURPOSE:
 *
 * This definitive version combines all advanced features: the CNN-LSTM
 * architecture, a three-class prediction target, and a robust
 * Walk-Forward Validation framework.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 26, 2025
 *
 * VERSION:             60.0 (Ultimate V2)
 *
 ******************************************************************************/
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

import config as config
from data_handler import DataHandler
from core.market_intelligence import MarketIntelligence

# --- 1. CONFIGURATION ---
SYMBOL = 'EURUSD'
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
SEQUENCE_LENGTH = 60
EPOCHS = 100
PREDICTION_HORIZON = 120
FLAT_THRESHOLD_ATR_FACTOR = 0.5

def create_training_data():
    """ Fetches data and engineers features with a three-class target. """
    print("--- Stage 1: Creating Training Data ---")
    data_handler = DataHandler(config)
    market_intel = MarketIntelligence(data_handler, config)
    data_handler.connect()
    timezone = pytz.timezone("Etc/UTC")
    start_date_dt = datetime.strptime(START_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    end_date_dt = datetime.strptime(END_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    df = data_handler.get_data_by_range(SYMBOL, TIMEFRAME, start_date_dt, end_date_dt)
    data_handler.disconnect()
    if df is None or df.empty:
        print("Could not fetch data. Aborting.")
        return None
    print("Calculating and engineering features...")
    df_features = market_intel._analyze_data(df).copy()
    
    # Advanced Target Engineering
    print("Engineering three-class target labels...")
    future_price = df_features['Close'].shift(-PREDICTION_HORIZON)
    price_change = future_price - df_features['Close']
    threshold = df_features['ATRr_14'] * FLAT_THRESHOLD_ATR_FACTOR
    
    df_features['target'] = 0 # Default to Flat
    df_features.loc[price_change > threshold, 'target'] = 1 # Up
    df_features.loc[price_change < -threshold, 'target'] = 2 # Down
    
    df_features.dropna(inplace=True)
    print(f"Feature engineering complete. Dataset has {len(df_features)} bars.")
    return df_features

def build_model(input_shape, num_classes):
    """ Builds the advanced CNN-LSTM model architecture for multi-class classification. """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def walk_forward_validation(df):
    """ Performs walk-forward validation on the three-class time-series data. """
    print("\n--- Stage 2: Performing Walk-Forward Validation with Three-Class CNN-LSTM Model ---")
    
    features = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'hurst'])
    target = df['target'].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        X.append(scaled_features[i-SEQUENCE_LENGTH:i])
        y.append(target[i])
    X, y = np.array(X), np.array(y)
    
    # One-Hot Encode the target labels for the multi-class model
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    num_classes = y_encoded.shape[1]

    n_train = int(len(X) * 0.6)
    n_test = int(len(X) * 0.1)
    
    predictions, actuals = [], []
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    for i in range(n_train, len(X), n_test):
        train_end, test_end = i, min(i + n_test, len(X))
        X_train, y_train_encoded = X[0:train_end], y_encoded[0:train_end]
        X_test, y_test_encoded = X[train_end:test_end], y_encoded[train_end:test_end]
        
        if len(X_test) == 0: continue

        print(f"\nTraining fold: Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
        
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
        
        # Calculate class weights on the original (non-encoded) labels for the current fold
        y_train_labels = np.argmax(y_train_encoded, axis=1)
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weights = dict(enumerate(weights))
        
        model.fit(X_train, y_train_encoded, epochs=EPOCHS, batch_size=32, class_weight=class_weights, verbose=1,
                  validation_split=0.1, callbacks=[early_stopping, reduce_lr])
        
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_labels = np.argmax(y_test_encoded, axis=1)
        predictions.extend(y_pred)
        actuals.extend(y_test_labels)

    print("\n\n--- Overall Walk-Forward Validation Results ---")
    print("\nClassification Report:")
    print(classification_report(actuals, predictions, target_names=['Flat', 'Up', 'Down']))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(actuals, predictions)
    print(cm)
    
    print("\nTraining final model on all available data...")
    final_model = build_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
    y_labels = np.argmax(y_encoded, axis=1)
    final_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
    final_class_weights = dict(enumerate(final_weights))
    final_model.fit(X, y_encoded, epochs=EPOCHS, batch_size=32, class_weight=final_class_weights, verbose=1,
                    callbacks=[early_stopping, reduce_lr], validation_split=0.1)
    final_model.save(f"model_{SYMBOL}_ultimate.h5")
    print(f"\nFinal ultimate model saved to 'model_{SYMBOL}_ultimate.h5'")

if __name__ == '__main__':
    training_df = create_training_data()
    if training_df is not None:
        walk_forward_validation(training_df)