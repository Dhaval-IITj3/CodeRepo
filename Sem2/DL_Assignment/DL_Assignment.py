import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from keras.optimizers import Adam

data = pd.read_csv('Health_data.csv')
data.info()