import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LSTM, SimpleRNN

def split_sequence(sequence, step):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_index = i + step
        if end_index > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_index], sequence[end_index]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


x = [i for i in np.arange(start=-10, stop=10, step=0.1)]
train_y = [np.sin(i) for i in x]

n_timesteps = 15
n_features = 1
train_x, train_y = split_sequence(train_y, n_timesteps)
print('shape x: {} / y: {}'.format(train_x.shape, train_y.shape))

train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
print('train_x.shape = ', train_x.shape)
print('train_y.shpae = ', train_y.shape)

model = Sequential()
model.add(SimpleRNN(units=10, return_sequences=False, input_shape=(n_timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

np.random.seed(0)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
history = model.fit(train_x, train_y, epochs = 1000, callbacks=[early_stopping])

plt.plot(history.history['loss'], label='loss')
plt.legend(loc='upper right')
plt.show()

test_x = np.arange(10, 40, 0.1)
calc_y = np.cos(test_x)

test_y = calc_y[:n_timesteps]
for i in range(len(test_x) - n_timesteps):
    net_input  = test_y[i : i + n_timesteps]
    net_input = net_input.reshape((1, n_timesteps, n_features))
    train_y = model.predict(net_input, verbose=0)
    print(test_y.shape, train_y.shape, i, i + n_timesteps)
    test_y = np.append(test_y, train_y)
    #test_y = np.append(test_y, model.predict(net_input)[0])

plt.plot(test_x, calc_y, label='ground truth', color='orange')
plt.plot(test_x, test_y, label='predictions', color='blue')
plt.legend(loc='upper left')
plt.ylim(-2,2)
plt.show()
