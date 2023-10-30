import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def create_dataset(X, y, lookback=1):
    dataX, dataY = [], []
    for i in range(len(X) - lookback):
        dataX.append(X[i:(i + lookback), :])
        dataY.append(y[i + lookback, 0])
    return np.array(dataX), np.array(dataY)

# Load datasets
data = pd.read_csv("data.csv")
X = data[['SOFR', 'EFFR', 'FFF', 'inflation exp', 'unemployment']][956:].values 
y = data[['SOFR_t+1']][956:].values

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data into training, validation, and test sets (60-20-20 split)
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size
train_X, train_y = X_scaled[:train_size], y_scaled[:train_size]
val_X, val_y = X_scaled[train_size:train_size + val_size], y_scaled[train_size:train_size + val_size]
test_X, test_y = X_scaled[train_size + val_size:], y_scaled[train_size + val_size:]

# Define lookback period (20 days)
lookback = 20

# Create data windows using the lookback period
train_X, train_y = create_dataset(train_X, train_y, lookback)
val_X, val_y = create_dataset(val_X, val_y, lookback)
test_X, test_y = create_dataset(test_X, test_y, lookback)

# Reshape input data
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 5))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 5))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 5))

# Import the tuning toolkit 
from hyperparameter_tuning import build_model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='tuning_results',
    project_name='lstm_time_series'
)

# Define early stopping criteria
early_stop = EarlyStopping(monitor='val_loss', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

tuner.search(train_X, train_y, epochs=10, validation_data=(val_X, val_y), callbacks=[early_stop, reduce_lr])

best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(train_X, train_y, epochs=300, batch_size=4, verbose=2, validation_data=(val_X, val_y), callbacks=[early_stop, reduce_lr, checkpoint])

# Plot training and validation loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Predictions
trainPredict = best_model.predict(train_X)
valPredict = best_model.predict(val_X)
testPredict = best_model.predict(test_X)

# Inverse normalization
trainPredict = scaler_y.inverse_transform(trainPredict)
valPredict = scaler_y.inverse_transform(valPredict)
testPredict = scaler_y.inverse_transform(testPredict)

# Convert train_y, val_y, and test_y to 2D arrays
train_y = train_y.reshape(-1, 1)
val_y = val_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

train_y = scaler_y.inverse_transform(train_y)
val_y = scaler_y.inverse_transform(val_y)
test_y = scaler_y.inverse_transform(test_y)

# Calculate RMSE
trainScore = np.sqrt(mean_squared_error(train_y, trainPredict))
print('Train RMSE: %.4f' % (trainScore))
valScore = np.sqrt(mean_squared_error(val_y, valPredict))
print('Validation RMSE: %.4f' % (valScore))
testScore = np.sqrt(mean_squared_error(test_y, testPredict))
print('Test RMSE: %.4f' % (testScore))