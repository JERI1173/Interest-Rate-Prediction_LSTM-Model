# Interest-Rate-Prediction_LSTM-Model

This project aims to predict future values of the SOFR (Secured Overnight Financing Rate) using historical data and various economic indicators. The primary features used for prediction include historical values of SOFR, EFFR (Effective Federal Funds Rate), FFF (Federal Funds Futures), inflation expectations, and unemployment rates.

## A. Hyperparameter
### Defining the Model with Hyperparameters:
The build_model function defines the architecture of the LSTM model. Within this function, hyperparameters are specified using the hp object. For instance, hp.Int('lstm_units_1', 30, 60, 10) indicates that the number of units in the first LSTM layer can vary between 30 and 60 in steps of 10
### Setting up the Tuner:
The RandomSearch class from keras-tuner is used to perform the hyperparameter search. It requires the model-building function (build_model), the optimization objective (val_loss in this case), and other parameters like the maximum number of trials.
### Search for Best Hyperparameters:
The tuner.search method initiates the hyperparameter search. It trains the model multiple times with different combinations of hyperparameters and validates the performance. Callbacks like early stopping and learning rate reduction are used during this process.