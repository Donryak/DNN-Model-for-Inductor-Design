import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define frequency bands
frequency_bands = [100, 200, 300, 400, 500, 600, 700, 800, 900]

# Load datasets
dataframes = {}
for freq in frequency_bands:
    file_path = f'C:/Users/HOME/Desktop/Program/DATASET/frequency_{freq}.0MHz.csv'
    dataframes[freq] = pd.read_csv(file_path)
    print(f"Loaded data for {freq} MHz")

# Print loaded data
for freq, df in dataframes.items():
    print(f"Data for {freq} MHz:")
    print(df.head())

def create_and_train_model(data, freq):
    # 데이터 분할
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train = scaler_X.fit_transform(train_data[['L', 'Q']].values)
    Y_train = scaler_Y.fit_transform(train_data[['N', 'L1', 'L2', 'W', 'S']].values)
    X_test = scaler_X.transform(test_data[['L', 'Q']].values)
    Y_test = scaler_Y.transform(test_data[['N', 'L1', 'L2', 'W', 'S']].values)

    # 신경망 모델 설계
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Input shape 수정
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(5, activation='linear')
    ])

    # 조기 종료와 학습률 감소 콜백 추가
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber(), metrics=['mae', 'mse'])

    # 모델 훈련
    history = model.fit(X_train, Y_train, epochs=200, batch_size=16, validation_data=(X_test, Y_test), verbose=1, callbacks=[reduce_lr, early_stopping])

    # 훈련 손실과 검증 손실 시각화 및 저장
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / MAE / MSE')
    plt.legend()
    plt.title(f'{freq}.0MHz_Model Metrics')
    loss_plot_path = f'C:/Users/HOME/Desktop/Program/DNN MODEL/{freq}.0MHz_metrics_plot.png'
    plt.savefig(loss_plot_path)

    # 최종 손실 값 출력
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    final_train_mse = history.history['mse'][-1]
    final_val_mse = history.history['val_mse'][-1]
    
    print(f"Final Training Loss: {final_train_loss}")
    print(f"Final Validation Loss: {final_val_loss}")
    print(f"Final Training MAE: {final_train_mae}")
    print(f"Final Validation MAE: {final_val_mae}")
    print(f"Final Training MSE: {final_train_mse}")
    print(f"Final Validation MSE: {final_val_mse}")

    # 평가 지표를 데이터프레임으로 저장
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'mae': history.history['mae'],
        'val_mae': history.history['val_mae'],
        'mse': history.history['mse'],
        'val_mse': history.history['val_mse']
    })
    metrics_file_path = f'C:/Users/HOME/Desktop/Program/DNN MODEL/{freq}.0MHz_metrics.csv'
    metrics_df.to_csv(metrics_file_path, index=False)

    # 모델 및 스케일러 저장
    model_save_path = f'C:/Users/HOME/Desktop/Program/DNN MODEL/TR_model_{freq}.0MHz.h5'
    scaler_X_save_path = f'C:/Users/HOME/Desktop/Program/DNN MODEL/scaler_X_{freq}.0MHz.pkl'
    scaler_Y_save_path = f'C:/Users/HOME/Desktop/Program/DNN MODEL/scaler_Y_{freq}.0MHz.pkl'

    model.save(model_save_path)
    joblib.dump(scaler_X, scaler_X_save_path)
    joblib.dump(scaler_Y, scaler_Y_save_path)

    # 저장 확인
    if os.path.exists(model_save_path) and os.path.exists(scaler_X_save_path) and os.path.exists(scaler_Y_save_path) and os.path.exists(loss_plot_path) and os.path.exists(metrics_file_path):
        print(f"Model, scalers, plot, and metrics for {freq} MHz are saved successfully!")
    else:
        print(f"Saving failed for {freq} MHz.")

    return model, scaler_X, scaler_Y

# Iterate over frequency bands
models = {}
scalers_X = {}
scalers_Y = {}

for freq, data in dataframes.items():
    model, scaler_X, scaler_Y = create_and_train_model(data, freq)
    models[freq] = model
    scalers_X[freq] = scaler_X
    scalers_Y[freq] = scaler_Y

print("All models and scalers have been saved successfully.")
