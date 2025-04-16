import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Dense
import os
import numpy as np

def run_dl_model():
    df = pd.read_csv("Cleaned-Data-Final.csv")
    X = df.drop('output', axis=1)
    y = df['output']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_file = "dl_model.h5"
    if not os.path.exists(model_file):
        dl_model = Sequential()
        dl_model.add(Dense(500, input_dim=X_train.shape[1], activation='relu'))
        dl_model.add(Dense(100, activation='relu'))
        dl_model.add(Dense(50, activation='relu'))
        dl_model.add(Dense(1, activation='sigmoid'))
        dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        dl_model.fit(X_train, y_train, epochs=50, verbose=0)
        dl_model.save(model_file)
    else:
        dl_model = load_model(model_file)

    y_pred_dl = dl_model.predict(X_test)
    y_pred_dl = (y_pred_dl > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred_dl)
    return f"{acc * 100:.2f}%"
