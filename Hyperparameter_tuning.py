from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from kerastuner import HyperParameters, RandomSearch

def build_model(hp: HyperParameters):
    model = Sequential()
    model.add(Bidirectional(LSTM(hp.Int('lstm_units_1', 30, 60, 10), return_sequences=True), input_shape=(lookback, 5)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(hp.Int('lstm_units_2', 20, 40, 10), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model