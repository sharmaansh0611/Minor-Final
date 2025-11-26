"""
tinyml_train.py
Train a tiny classifier (Keras) to detect 'event' vs 'no-event' using CSV traces.
Exports a quantized TFLite model suitable for microcontroller deployment.
"""
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    # Create a label: wake_reason == 'event' -> 1 else 0
    df['label'] = (df['wake_reason'] == 'event').astype(int)
    # features: sensor0, sensor1, battery_pct, rssi, recent rolling event rate approximated by sensor1
    df['sensor1_rolling'] = df['sensor1'].rolling(window=5, min_periods=1).mean()
    features = df[['sensor0','sensor1','sensor1_rolling','battery_pct','rssi']].fillna(0.0).values
    labels = df['label'].values
    return features, labels

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def convert_to_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote TFLite model to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/dummy_traces.csv")
    parser.add_argument("--out", default="training/models/tiny_model.tflite")
    args = parser.parse_args()

    X, y = load_features(args.input)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    # Save Keras model optionally
    os.makedirs("training/models", exist_ok=True)
    model.save("training/models/tiny_model.h5")
    convert_to_tflite(model, args.out)

if __name__ == "__main__":
    main()
