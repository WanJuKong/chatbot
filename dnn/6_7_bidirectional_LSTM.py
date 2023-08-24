import numpy as np
from random import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed

def get_sequence(n_timesteps):
    X = np.array([random() for _ in range(n_timesteps)])
    limit= n_timesteps / 4.0
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)]) #[x<limit?0:1; for x in cumsum]

    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y

n_units = 20
n_timesteps = 4

model = Sequential()
model.add(Bidirectional(LSTM(n_units, return_sequences=True, input_shape=(n_timesteps, 1))))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in range(1000):
    X, y = get_sequence(n_timesteps)
    model.fit(X, y, epochs = 1, batch_size = 1, verbose = 2)


y_pred = model.predict(X, verbose=0)
yhat = (y_pred > 0.5).astype(int)

for i in range(n_timesteps):
    print('실제 값: {}, 예측 값: {}'.format(y[0, i], yhat[0, i]))

#X, y = model.predict(X, verbose = 0)
#for i in range(n_timesteps):
#    print('실제 값: {}, 예측 값: {}'.format(y[0, i], yhat[0, i]))
